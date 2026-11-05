"""Histogram sampler used to feed NeuralOT (which expects continuous samplers) with a discrete histogram: a bin is
drawn with its probability, then a uniform point inside it."""
import torch


class HistogramSampler:
    """
    Sampler that draws samples from a 1D histogram.
    Used to interface discrete dataset distributions with Neural OT code which expects continuous samplers.
    """
    def __init__(self, histogram, ranges=(0, 1), device='cuda'):
        self.histogram = histogram / histogram.sum()
        self.ranges = ranges
        self.device = device
        self.n_bins = len(histogram)
        self.bin_edges = torch.linspace(ranges[0], ranges[1], self.n_bins + 1, device=device)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
    def sample(self, batch_size):
        # Sample bins based on histogram probabilities
        bin_indices = torch.multinomial(self.histogram, batch_size, replacement=True)
        
        # Add uniform noise within the bin to make it continuous-ish
        # Or just return bin centers? NOT typically expects continuous samples.
        # Let's return uniform samples within the selected bins.
        
        bin_width = self.bin_edges[1] - self.bin_edges[0]
        offsets = torch.rand(batch_size, device=self.device) * bin_width
        
        lower_edges = self.bin_edges[bin_indices]
        samples = lower_edges + offsets
        
        return samples.view(-1, 1) # (B, 1)
