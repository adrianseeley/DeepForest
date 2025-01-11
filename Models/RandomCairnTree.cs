public class RandomCairnTree
{
    public enum Minimize
    {
        MeanSquaredError,
        MeanAbsoluteError
    };

    public float[] output;
    public float error;
    public float[]? leftCairn;
    public float[]? rightCairn;
    public RandomCairnTree? left;
    public RandomCairnTree? right;

    public static RandomCairnTree Build(List<Sample> samples, int minSamplesPerLeaf, int maxLeafDepth, Minimize minimize)
    {
        float[,] sampleDistances = new float[samples.Count, samples.Count];
        for (int a = 0; a < samples.Count; a++)
        {
            for (int b = a; b < samples.Count; b++)
            {
                float distance = Utility.EuclideanDistance(samples[a].input, samples[b].input);
                sampleDistances[a, b] = distance;
                sampleDistances[b, a] = distance;
            }
        }
        List<int> sampleIndices = Enumerable.Range(0, samples.Count).ToList();
        return new RandomCairnTree(samples, sampleIndices, sampleDistances, minSamplesPerLeaf, maxLeafDepth, minimize);
    }

    private RandomCairnTree(List<Sample> samples, List<int> sampleIndices, float[,] sampleDistances, int minSamplesPerLeaf, int maxLeafDepth, Minimize minimize, int currentDepth = 0)
    {
        output = CalculateAverageOutput(samples, sampleIndices);
        error = CalculateError(samples, sampleIndices, output, minimize);
        leftCairn = null;
        rightCairn = null;
        left = null;
        right = null;
        if (currentDepth >= maxLeafDepth || sampleIndices.Count < minSamplesPerLeaf * 2)
        {
            return;
        }

        // shuffle indices
        Random random = new Random();
        sampleIndices = sampleIndices.OrderBy(i => random.Next()).ToList();

        int bestLeftCairnIndex = -1;
        int bestRightCairnIndex = -1;
        List<int> leftIndices = new List<int>(sampleIndices.Count);
        List<int> rightIndices = new List<int>(sampleIndices.Count);
        bool splitFound = false;
        for (int a = 0; a < sampleIndices.Count; a++)
        {
            int leftCairnIndex = sampleIndices[a];
            for (int b = a + 1; b < sampleIndices.Count; b++)
            {
                int rightCairnIndex = sampleIndices[b];
                leftIndices.Clear();
                rightIndices.Clear();
                foreach (int sampleIndex in sampleIndices)
                {
                    if (sampleDistances[leftCairnIndex, sampleIndex] <= sampleDistances[rightCairnIndex, sampleIndex])
                    {
                        leftIndices.Add(sampleIndex);
                    }
                    else
                    {
                        rightIndices.Add(sampleIndex);
                    }
                }
                if (leftIndices.Count >= minSamplesPerLeaf && rightIndices.Count >= minSamplesPerLeaf)
                {
                    bestLeftCairnIndex = leftCairnIndex;
                    bestRightCairnIndex = rightCairnIndex;
                    splitFound = true;
                    break;
                }
            }
            if (splitFound)
            {
                break;
            }
        }

        if (bestLeftCairnIndex != -1)
        {
            leftCairn = samples[bestLeftCairnIndex].input;
            rightCairn = samples[bestRightCairnIndex].input;
            left = new RandomCairnTree(samples, leftIndices, sampleDistances, minSamplesPerLeaf, maxLeafDepth, minimize, currentDepth + 1);
            right = new RandomCairnTree(samples, rightIndices, sampleDistances, minSamplesPerLeaf, maxLeafDepth, minimize, currentDepth + 1);
        }
    }

    public float[] Predict(float[] input)
    {
        if (left == null || right == null || leftCairn == null || rightCairn == null)
        {
            return output;
        }
        if (Utility.EuclideanDistance(input, leftCairn) <= Utility.EuclideanDistance(input, rightCairn))
        {
            return left.Predict(input);
        }
        else
        {
            return right.Predict(input);
        }
    }

    private static float[] CalculateAverageOutput(List<Sample> samples, List<int> sampleIndices)
    {
        float[] averageOutput = new float[samples[0].output.Length];
        foreach (int sampleIndex in sampleIndices)
        {
            Sample sample = samples[sampleIndex];
            for (int i = 0; i < sample.output.Length; i++)
            {
                averageOutput[i] += sample.output[i];
            }
        }
        for (int i = 0; i < averageOutput.Length; i++)
        {
            averageOutput[i] /= sampleIndices.Count;
        }
        return averageOutput;
    }

    private static float CalculateError(List<Sample> samples, List<int> sampleIndices, float[] averageOutput, Minimize minimize)
    {
        float error;
        switch (minimize)
        {
            case Minimize.MeanSquaredError:
                error = 0f;
                foreach(int sampleIndex in sampleIndices)
                {
                    Sample sample = samples[sampleIndex];
                    for (int i = 0; i < sample.output.Length; i++)
                    {
                        float difference = sample.output[i] - averageOutput[i];
                        error += difference * difference;
                    }
                }
                return error / sampleIndices.Count;
            case Minimize.MeanAbsoluteError:
                error = 0f;
                foreach (int sampleIndex in sampleIndices)
                {
                    Sample sample = samples[sampleIndex];
                    for (int i = 0; i < sample.output.Length; i++)
                    {
                        error += Math.Abs(sample.output[i] - averageOutput[i]);
                    }
                }
                return error / sampleIndices.Count;
            default:
                throw new Exception("Unknown reduction.");
        }
    }
}