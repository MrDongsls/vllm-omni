# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from functools import wraps
from typing import Any, cast

import torch
from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_pp_group,
    is_pipeline_first_stage,
)


class AsyncLatents:
    """Transparent async wrapper returned by scheduler_step on rank 0.

    Wraps a pending ``irecv_tensor_dict`` and defers ``handle.wait()`` until the
    underlying tensor is actually consumed — either via attribute access
    (e.g. ``latents.to(dtype)``, ``latents.shape``) or via a torch operation
    (e.g. ``mask * latents``).  This keeps the first PP rank non-blocking after
    posting the receive, matching the async philosophy used everywhere else in
    the PP communication layer.
    """

    __slots__ = ("_tensor_dict", "_handles", "_postproc", "_tensor")

    def __init__(
        self,
        tensor_dict: dict[str, torch.Tensor],
        handles: list[torch.distributed.Work],
        postproc: list,
    ):
        self._tensor_dict = tensor_dict
        self._handles = handles
        self._postproc = postproc
        self._tensor: torch.Tensor | None = None

    def _resolve(self) -> torch.Tensor:
        if self._tensor is not None:
            return self._tensor
        for h in self._handles:
            h.wait()
        for fn in self._postproc:
            fn()
        self._tensor = self._tensor_dict["latents"]
        return self._tensor

    # Attribute access (e.g. .shape, .to(), .dtype) delegates to the resolved tensor.
    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    # Torch function protocol: any torch op involving an AsyncLatents resolves it first.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        def _unwrap(x):
            if isinstance(x, AsyncLatents):
                return x._resolve()
            if isinstance(x, (list, tuple)):
                return type(x)(_unwrap(item) for item in x)  # type(x) return the class of x to preserve its type
            return x

        args = tuple(_unwrap(a) for a in args)
        kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)


class PipelineParallelMixin:
    """
    Mixin providing Pipeline Parallelism for diffusion pipelines.

    All PP ranks run the full denoising loop in `forward()`.
    `predict_noise_maybe_with_cfg` and `scheduler_step_maybe_with_cfg` encapsulate all inter-rank communication.

    Communication pattern per denoising step:
      Forward chain : rank 0 → 1 → … → N-1  via async isend/irecv (AsyncIntermediateTensors)
      Next timestep : last rank → rank 0     via async isend/irecv (AsyncLatents)

    All communication is asynchronous using isend_tensor_dict/irecv_tensor_dict.
    Only rank 0 needs updated latents for the next forward pass start.

    For sequential CFG (cfg_parallel_size=1) with PP, two full forward chains are
    executed — one for the positive pass and one for the negative pass — so that each
    PP stage operates on the correct encoder_hidden_states.
    """

    vae: Any

    @staticmethod
    def _slice_pp_prediction(
        prediction: torch.Tensor | tuple[torch.Tensor, ...],
        output_slice: int,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(prediction, torch.Tensor):
            return prediction[:, :output_slice]
        if isinstance(prediction, tuple) and all(isinstance(item, torch.Tensor) for item in prediction):
            return tuple(item[:, :output_slice] for item in prediction)
        raise TypeError("PP predictions must be tensors or tuples of tensors")

    @staticmethod
    def _pp_world_size_or_one() -> int:
        """Keep direct, non-distributed pipeline use on the PP=1 path."""
        try:
            return get_pipeline_parallel_world_size()
        except AssertionError:
            return 1

    def _cfg_collect_on_this_rank(self) -> bool:
        """Only the last PP stage has predictions to exchange or combine."""
        if self._pp_world_size_or_one() == 1:
            return True
        return get_pp_group().is_last_rank

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        init = cls.__dict__.get("__init__")
        if callable(init):

            @wraps(init)
            def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
                init(self, *args, **kwargs)
                vae = getattr(self, "vae", None)
                if vae is not None and hasattr(vae, "decode"):
                    self._wrapped_vae_decode()

            setattr(cls, "__init__", wrapped_init)

        diffuse = cls.__dict__.get("diffuse")
        if callable(diffuse):

            @wraps(diffuse)
            def wrapped_diffuse(self, *args: Any, **kwargs: Any) -> Any:
                try:
                    latents = diffuse(self, *args, **kwargs)
                    if isinstance(latents, AsyncLatents):
                        latents = torch.as_tensor(latents)  # avoid copying
                    return latents
                finally:
                    self._sync_pp_send()

            setattr(cls, "diffuse", wrapped_diffuse)

    def _wrapped_vae_decode(self) -> None:
        vae, orig_decode = self.vae, self.vae.decode

        @wraps(orig_decode)
        def wrapped_decode(z: torch.Tensor, *args: Any, **kwargs: Any):
            if hasattr(vae, "is_distributed_enabled") and vae.is_distributed_enabled():
                # Middle ranks (world size 3 or more) hold stale latents after the denoising loop.
                # Broadcast from rank 0 so every rank splits identical tiles.
                if get_pipeline_parallel_world_size() > 2:
                    z = get_pp_group().broadcast(z, src=0)
                return orig_decode(z, *args, **kwargs)
            elif is_pipeline_first_stage():
                return orig_decode(z, *args, **kwargs)
            return (None,)  # decoder returns a tuple

        self.vae.decode = wrapped_decode

    @property
    def _pp_send_work(self) -> list[torch.distributed.Work]:
        if not hasattr(self, "_pp_send_work_list"):
            self._pp_send_work_list: list[torch.distributed.Work] = []
        return self._pp_send_work_list

    @_pp_send_work.setter
    def _pp_send_work(self, work: list[torch.distributed.Work]) -> None:
        self._pp_send_work_list = work

    def _sync_pp_send(self) -> None:
        """
        Wait on all pending non-blocking PP sends.

        Must be called after the denoising loop so that the isend handles
        from the last iteration are completed before any subsequent
        collective (e.g. VAE decode broadcast) or tensor reuse.
        """
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

    def _run_branch(
        self,
        kwargs: dict[str, Any],
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Run one branch through the PP stages for multi-branch CFG."""
        if self._pp_world_size_or_one() == 1:
            return cast(Any, self).predict_noise(**kwargs)

        self._sync_pp_send()
        pp_group = get_pp_group()
        intermediate = None
        if not pp_group.is_first_rank:
            intermediate = AsyncIntermediateTensors(*pp_group.irecv_tensor_dict())

        if not pp_group.is_last_rank:
            result = cast(Any, self).predict_noise(**kwargs, intermediate_tensors=intermediate)
            self._pp_send_work.extend(pp_group.isend_tensor_dict(result.tensors))
            return None

        return cast(Any, self).predict_noise(**kwargs, intermediate_tensors=intermediate)

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        cfg_normalize: bool = True,
        output_slice: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """
        Drop-in replacement for predict_noise_maybe_with_cfg that also handles PP.

        Supports three modes:
          - PP only, sequential CFG: both branches (cond and uncond) run through this PP pipeline.
            This doubles communication volume per denoising step compared to PP + CFG-parallel.
          - PP + CFG-parallel: each PP pipeline carries one branch. The last PP
            rank all-gathers across the CFG group and combines, mirroring
            CFGParallelMixin.predict_noise_maybe_with_cfg exactly.
          - PP only, no CFG: cond branch only.

        Returns:
            noise_pred on the last PP rank (all CFG ranks when CFG-parallel is active).
            None on all other ranks.
        """
        if self._pp_world_size_or_one() == 1:
            return cast(Any, super()).predict_noise_maybe_with_cfg(
                do_true_cfg, true_cfg_scale, positive_kwargs, negative_kwargs, cfg_normalize, output_slice
            )

        self._sync_pp_send()

        pp_group = get_pp_group()

        cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1
        all_kwargs: list[dict[str, Any]]
        if cfg_parallel_ready:
            # Each PP pipeline carries exactly one CFG branch determined by cfg_rank.
            all_kwargs = [
                positive_kwargs if get_classifier_free_guidance_rank() == 0 else cast(dict[str, Any], negative_kwargs)
            ]
        else:
            # Sequential CFG (or no CFG): this PP pipeline handles all branches.
            all_kwargs = [positive_kwargs]
            if do_true_cfg:
                all_kwargs.append(cast(dict[str, Any], negative_kwargs))

        # Non-first ranks receive intermediate tensors asynchronously
        n = len(all_kwargs)
        its: list[AsyncIntermediateTensors | None] = [None] * n
        if not pp_group.is_first_rank:
            for i in range(n):
                its[i] = AsyncIntermediateTensors(*pp_group.irecv_tensor_dict())

        if not pp_group.is_last_rank:
            # First / middle rank: run partial forwards and propagate ITs downstream.
            for kwargs, it in zip(all_kwargs, its):
                result = cast(Any, self).predict_noise(**kwargs, intermediate_tensors=it)
                self._pp_send_work.extend(pp_group.isend_tensor_dict(result.tensors))
            return None

        # Last rank: run full forward
        noise_preds = [
            cast(Any, self).predict_noise(**kwargs, intermediate_tensors=it) for kwargs, it in zip(all_kwargs, its)
        ]

        if cfg_parallel_ready:
            # All-gather the single-branch prediction across the CFG group and combine
            # on all CFG ranks so every last PP rank has an identical noise_pred.
            local_pred = noise_preds[0]
            if output_slice is not None:
                local_pred = self._slice_pp_prediction(local_pred, output_slice)
            local_predictions = local_pred if isinstance(local_pred, tuple) else (local_pred,)
            gathered = [get_cfg_group().all_gather(pred, separate_tensors=True) for pred in local_predictions]
            positive = tuple(per_element[0] for per_element in gathered)
            negative = tuple(per_element[1] for per_element in gathered)
            if not isinstance(local_pred, tuple):
                positive, negative = positive[0], negative[0]
            return cast(Any, self).combine_cfg_noise(positive, negative, true_cfg_scale, cfg_normalize)

        # Sequential CFG or no-CFG path.
        if do_true_cfg:
            pos, neg = noise_preds[0], noise_preds[1]
            if output_slice is not None:
                pos = self._slice_pp_prediction(pos, output_slice)
                neg = self._slice_pp_prediction(neg, output_slice)
            return cast(Any, self).combine_cfg_noise(pos, neg, true_cfg_scale, cfg_normalize)
        pred = noise_preds[0]
        if output_slice is not None:
            pred = self._slice_pp_prediction(pred, output_slice)
        return pred

    def scheduler_step_maybe_with_cfg(
        self,
        noise_pred: torch.Tensor | tuple[torch.Tensor, ...] | None,
        t: torch.Tensor | tuple[torch.Tensor, ...],
        latents: torch.Tensor | tuple[torch.Tensor, ...],
        do_true_cfg: bool,
        per_request_scheduler: Any | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | AsyncLatents:
        """
        Drop-in replacement for scheduler_step_maybe_with_cfg that also handles PP.

        Only the last rank runs the scheduler (it already has noise_pred); the result
        is sent to rank 0 which needs it for the next forward pass.

        Returns a ``AsyncLatents`` on rank 0 that transparently defers
        ``handle.wait()`` until the tensor is actually consumed (via attribute
        access or a torch operation), keeping the rank non-blocking after the
        ``irecv`` is posted.
        """
        if self._pp_world_size_or_one() == 1:
            return cast(Any, super()).scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg, per_request_scheduler, generator
            )

        pp_group = get_pp_group()
        if pp_group.is_last_rank:
            latents = cast(Any, super()).scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg, per_request_scheduler, generator
            )
            self._pp_send_work = pp_group.isend_tensor_dict({"latents": latents}, dst=0)
        elif pp_group.is_first_rank:
            latents = AsyncLatents(*pp_group.irecv_tensor_dict(src=pp_group.world_size - 1))
        return latents
