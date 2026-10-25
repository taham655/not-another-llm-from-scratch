import torch
import torch.nn as nn 

# a Linear layer is basically a transformation inputs into the desired outputs , 
# so if its Linear(2,3) -> linear would help us transform the 2 inputs into 3 outputs 
# while tracking the gradients
class Linear(nn.Module):
    def __init__(self, input_features, output_features, device=None, dtype=None):
        super().__init__()
        # pytorch's tensor convention is output first since outputs are represented by a row 
        self.weight = nn.Parameter(torch.empty(output_features, input_features, device=device, dtype=dtype))
        # start small: std = sqrt(2 / (in + out)), so outputs stay about the same
        # size as inputs instead of growing layer after layer. Truncated at +-3 std
        # so no weight starts out as a wild outlier.
        std = (2 / (input_features + output_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        # x is (batch, in) and weight is (out, in), so we transpose
        # weight to (in, out) to get (batch, in) @ (in, out) -> (batch, out)
        return x @ self.weight.T


if __name__ == "__main__":
    # The idea: predict 3 things about a student from 2 things we know.
    #   Inputs  (2): hours studied, hours slept
    #   Outputs (3): math score, english score, stress level
    layer = Linear(input_features=2, output_features=3)

    # A batch with 1 student: studied 5 hours, slept 8 hours -> shape (1, 2)
    # (floats, because the layer's weights are floats)
    student_batch = torch.tensor([[5.0, 8.0]])

    predictions = layer(student_batch)
    print("predictions:",predictions)
    math_score, english_score, stress_level = predictions[0]

    # The layer is untrained (random weights), so these numbers are
    # meaningless for now. Training is what would make them sensible.
    print("Math score:   ", math_score.item())
    print("English score:", english_score.item())
    print("Stress level: ", stress_level.item())
    print("Shape (batch, outputs):", predictions.shape)  # torch.Size([1, 3])

