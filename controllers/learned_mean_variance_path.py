from .mean_variance_path import Controller as MeanVariancePathController


class Controller(MeanVariancePathController):
  """Online path optimizer backed by the learned blackbox response model."""

  def __init__(self):
    super().__init__(use_learned_model=True)
