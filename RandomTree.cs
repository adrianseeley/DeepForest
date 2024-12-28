public class RandomTree
{
    public int yComponent;
    public float y;
    public float error;
    public int? splitXComponent;
    public float? splitXValue;
    public RandomTree? left;
    public RandomTree? right;

    public RandomTree(List<(List<float> input, List<float> output)> samples, List<int> xComponents, int yComponent)
    {
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }
        this.yComponent = yComponent;
        y = samples.Select(s => s.output[yComponent]).Average();
        error = samples.Select(s => Math.Abs(s.output[yComponent] - y)).Sum() / samples.Count;
        splitXComponent = null;
        splitXValue = null;
        left = null;
        right = null;
        float bestError = error;
        int bestXComponent = -1;
        float bestXValue = float.NaN;
        List<(List<float> input, List<float> output)> leftSamples;
        List<(List<float> input, List<float> output)> rightSamples;
        foreach (int xComponent in xComponents)
        {
            HashSet<float> xValues = new HashSet<float>(samples.Select(s => s.input[xComponent]));
            foreach (float xValue in xValues)
            {
                leftSamples = samples.Where(s => s.input[xComponent] <= xValue).ToList();
                rightSamples = samples.Where(s => s.input[xComponent] > xValue).ToList();
                if (leftSamples.Count == 0 || rightSamples.Count == 0)
                {
                    continue;
                }
                float leftY = leftSamples.Select(s => s.output[yComponent]).Average();
                float rightY = rightSamples.Select(s => s.output[yComponent]).Average();
                float leftError = leftSamples.Select(s => Math.Abs(s.output[yComponent] - leftY)).Sum() / leftSamples.Count;
                float rightError = rightSamples.Select(s => Math.Abs(s.output[yComponent] - rightY)).Sum() / rightSamples.Count;
                float jointError = (leftError * (leftSamples.Count / samples.Count)) + (rightError * (rightSamples.Count / samples.Count));
                if (jointError < bestError)
                {
                    bestError = jointError;
                    bestXComponent = xComponent;
                    bestXValue = xValue;
                }
            }
        }
        if (bestXComponent == -1)
        {
            return;
        }
        splitXComponent = bestXComponent;
        splitXValue = bestXValue;
        leftSamples = samples.Where(s => s.input[bestXComponent] <= bestXValue).ToList();
        rightSamples = samples.Where(s => s.input[bestXComponent] > bestXValue).ToList();
        left = new RandomTree(leftSamples, xComponents, yComponent);
        right = new RandomTree(rightSamples, xComponents, yComponent);
    }

    public float Predict(List<float> input)
    {
        if (splitXComponent == null)
        {
            return y;
        }
        if (input[splitXComponent.Value] <= splitXValue)
        {
            if (left == null)
            {
                throw new InvalidOperationException("Left child must not be null.");
            }
            return left.Predict(input);
        }
        else
        {
            if (right == null)
            {
                throw new InvalidOperationException("Right child must not be null.");
            }
            return right.Predict(input);
        }
    }
}