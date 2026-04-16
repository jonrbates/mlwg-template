from transformers import TrainerCallback


class FreezingCallback(TrainerCallback):
    """Gradually unfreeze layers during training based on step thresholds.

    All params are in the optimizer from the start. This callback just
    toggles requires_grad — frozen params get zero gradients so the
    optimizer updates are no-ops for them.

    Args:
        schedule: list of (step, modules_to_unfreeze) tuples, e.g.:
            [
                (0,   ["pre_classifier", "classifier"]),
                (100, ["distilbert.transformer.layer.5"]),
                (500, ["distilbert.transformer.layer.4"]),
            ]
    """

    def __init__(self, schedule):
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self._last_applied = -1

    def _apply(self, model, step):
        model.requires_grad_(False)
        applied = []
        for t, mods in self.schedule:
            if t <= step:
                for mod_name in mods:
                    model.get_submodule(mod_name).requires_grad_(True)
                applied.append((t, mods))

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[step {step}] Applied freeze schedule up through: {applied}")
        print(f"[step {step}] Trainable params: {trainable:,}\n")
        self._last_applied = step

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Ensure the correct freeze state on resume (step may be > 0).
        self._apply(model, state.global_step)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step == self._last_applied:
            return

        for threshold, _ in self.schedule:
            if step == threshold:
                self._apply(model, step)
                break
