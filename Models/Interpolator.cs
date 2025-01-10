public static class Interpolator
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

    public static void ComputeExponential(List<Sample> train, float[,] normalizedDistances, int testIndex, float exponent, float[,] testPredictions)
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

    public static void ComputeHinge(List<Sample> train, float[,] normalizedDistances, int testIndex, float hingePoint, float closeWeight, float farWeight, float[,] testPredictions)
    {
        float weightSum = 0f;
        for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
        {
            Sample sample = train[trainIndex];
            float weight = normalizedDistances[testIndex, trainIndex];
            if (weight >= hingePoint)
            {
                weight *= closeWeight;
            }
            else
            {
                weight *= farWeight;
            }
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

    public static void ComputeReverseWalkingAverage(List<Sample> train, List<float[]> testInputs, float[,] testPredictions)
    {
        for (int testIndex = 0; testIndex < testInputs.Count; testIndex++)
        {
            float[] testInput = testInputs[testIndex];
            List<(Sample sample, float distance)> distances = new List<(Sample sample, float distance)>();
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                Sample sample = train[trainIndex];
                float distance = Utility.EuclideanDistance(testInput, sample.input);
                distances.Add((sample, distance));
            }

            // sort far to near
            distances.Sort((a, b) => b.distance.CompareTo(a.distance));

            // add first sample as base of prediction
            for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
            {
                testPredictions[testIndex, outputIndex] = distances[0].sample.output[outputIndex];
            }

            // iterate through remaining train samples
            for (int trainIndex = 1; trainIndex < train.Count; trainIndex++)
            {
                Sample sample = distances[trainIndex].sample;

                // add sample to prediction
                for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
                {
                    testPredictions[testIndex, outputIndex] += sample.output[outputIndex];
                }

                // divide prediction by 2
                for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
                {
                    testPredictions[testIndex, outputIndex] /= 2f;
                }
            }
        }
    }
}