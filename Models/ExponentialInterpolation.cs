public static class ExponentialInterpolation
{
    public static float[,] ComputeNormalizedDistances(List<Sample> train, List<float[]> test, bool verbose = false)
    {
        float[,] normalizedDistances = new float[test.Count, train.Count];
        int completed = 0;
        Parallel.For(0, test.Count, testIndex =>
        {
            float[] testInput = test[testIndex];
            float maxDistance = float.NegativeInfinity;
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                float distance = Utility.EuclideanDistance(testInput, train[trainIndex].input);
                if (distance > maxDistance)
                {
                    maxDistance = distance;
                }
                normalizedDistances[testIndex, trainIndex] = distance;
            }
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                float normalizedDistance = 1f - (normalizedDistances[testIndex, trainIndex] / maxDistance);
                if (normalizedDistance == 0f)
                {
                    normalizedDistance = 0.00001f;
                }
                normalizedDistances[testIndex, trainIndex] = normalizedDistance;
            }
            if (verbose)
            {
                int localCompleted = Interlocked.Increment(ref completed);
                if (localCompleted % 1000 == 0)
                {
                    Console.Write($"\rComputeNormalizedDistances {localCompleted + 1}/{test.Count}");
                }
            }
        });
        if (verbose)
        {
            Console.WriteLine();
        }
        return normalizedDistances;
    }

    public static void Compute(List<Sample> train, float[,] normalizedDistances, int testIndex, float exponent, float[,] testPredictions)
    {
        float weightSum = 0f;
        for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
        {
            Sample sample = train[trainIndex];
            float weight = normalizedDistances[testIndex, trainIndex];
            weight = MathF.Pow(weight, exponent);
            weightSum += weight;
            for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
            {
                testPredictions[testIndex, outputIndex] += sample.output[outputIndex] * weight;
            }
        }
        for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
        {
            testPredictions[testIndex, outputIndex] /= weightSum;
        }
    }
}