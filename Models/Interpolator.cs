public static class Interpolator
{
    public static float[,] ComputeDistances(List<Sample> train, List<float[]> test, bool verbose = false)
    {
        float[,] distances = new float[test.Count, train.Count];
        int completed = 0;
        Parallel.For(0, test.Count, testIndex =>
        {
            float[] testInput = test[testIndex];
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                float distance = Utility.EuclideanDistance(testInput, train[trainIndex].input);
                distances[testIndex, trainIndex] = distance;
            }
            if (verbose)
            {
                int localCompleted = Interlocked.Increment(ref completed);
                if (localCompleted % 1000 == 0)
                {
                    Console.Write($"\rComputeDistances {localCompleted + 1}/{test.Count}");
                }
            }
        });
        if (verbose)
        {
            Console.WriteLine();
        }
        return distances;
    }

    public static float[,] ComputeGaussians(List<Sample> train, List<float[]> test, float sigma, bool verbose = false)
    {
        float denominator = 2 * MathF.Pow(sigma, 2);
        float[,] gaussians = new float[test.Count, train.Count];
        int completed = 0;
        Parallel.For(0, test.Count, testIndex =>
        {
            float[] testInput = test[testIndex];
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                float distance = Utility.EuclideanDistance(testInput, train[trainIndex].input);
                float gaussian = MathF.Exp(-MathF.Pow(distance, 2) / denominator);
                gaussians[testIndex, trainIndex] = distance;
            }
            if (verbose)
            {
                int localCompleted = Interlocked.Increment(ref completed);
                if (localCompleted % 1000 == 0)
                {
                    Console.Write($"\rComputeGaussians {localCompleted + 1}/{test.Count}");
                }
            }
        });
        if (verbose)
        {
            Console.WriteLine();
        }
        return gaussians;
    }

    public static float[,] ComputeReciprocalDistances(List<Sample> train, List<float[]> test, bool verbose = false)
    {
        float[,] reciprocalDistances = new float[test.Count, train.Count];
        int completed = 0;
        Parallel.For(0, test.Count, testIndex =>
        {
            float[] testInput = test[testIndex];
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                float distance = 1f / (Utility.EuclideanDistance(testInput, train[trainIndex].input) + 0.0000001f);
                reciprocalDistances[testIndex, trainIndex] = distance;
            }
            if (verbose)
            {
                int localCompleted = Interlocked.Increment(ref completed);
                if (localCompleted % 1000 == 0)
                {
                    Console.Write($"\rComputeReciprocalDistances {localCompleted + 1}/{test.Count}");
                }
            }
        });
        if (verbose)
        {
            Console.WriteLine();
        }
        return reciprocalDistances;
    }

    public static float[,] ComputeWeightedReciprocalDistances(List<Sample> train, List<float[]> test, float weight, bool verbose = false)
    {
        float[,] weightedReciprocalDistances = new float[test.Count, train.Count];
        int completed = 0;
        Parallel.For(0, test.Count, testIndex =>
        {
            float[] testInput = test[testIndex];
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                float distance = 1f / (MathF.Pow(Utility.EuclideanDistance(testInput, train[trainIndex].input), weight) + 0.0000001f);
                weightedReciprocalDistances[testIndex, trainIndex] = distance;
            }
            if (verbose)
            {
                int localCompleted = Interlocked.Increment(ref completed);
                if (localCompleted % 1000 == 0)
                {
                    Console.Write($"\rComputeWeightedReciprocalDistances {localCompleted + 1}/{test.Count}");
                }
            }
        });
        if (verbose)
        {
            Console.WriteLine();
        }
        return weightedReciprocalDistances;
    }

    public static float[,] ComputeExponentialInverseNormalizedDistances(List<Sample> train, List<float[]> test, float exponent, bool verbose = false)
    {
        float[,] einds = new float[test.Count, train.Count];
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
                einds[testIndex, trainIndex] = distance;
            }
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                float eind = 1f - (einds[testIndex, trainIndex] / maxDistance);
                if (eind == 0f)
                {
                    eind = 0.00001f;
                }
                eind = MathF.Pow(eind, exponent);
                einds[testIndex, trainIndex] = eind;
            }
            if (verbose)
            {
                int localCompleted = Interlocked.Increment(ref completed);
                if (localCompleted % 1000 == 0)
                {
                    Console.Write($"\rComputeExponentialInverseNormalizedDistances {localCompleted + 1}/{test.Count}");
                }
            }
        });
        if (verbose)
        {
            Console.WriteLine();
        }
        return einds;
    }

    public static void WeightedCombine(List<Sample> train, float[,] weights, float[,] testPredictions)
    {
        Parallel.For(0, testPredictions.GetLength(0), testIndex =>
        {
            float weightSum = 0f;
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                Sample trainSample = train[trainIndex];
                float weight = weights[testIndex, trainIndex];
                weightSum += weight;
                for (int outputIndex = 0; outputIndex < trainSample.output.Length; outputIndex++)
                {
                    testPredictions[testIndex, outputIndex] += trainSample.output[outputIndex] * weight;
                }
            }
            for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
            {
                testPredictions[testIndex, outputIndex] /= weightSum;
            }
        });
    }

    public static void WeightedCombineK(List<Sample> train, float[,] distances, float[,] weights, float[,] testPredictions, int k)
    {
        Parallel.For(0, testPredictions.GetLength(0), testIndex =>
        {
            List<(Sample trainSample, float distance, float weight)> neighbours = new List<(Sample trainSample, float distance, float weight)>();
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                Sample trainSample = train[trainIndex];
                float distance = distances[testIndex, trainIndex];
                float weight = weights[testIndex, trainIndex];
                neighbours.Add((trainSample, distance, weight));
            }
            neighbours = neighbours.OrderBy(a => a.distance).Take(k).ToList();
            float weightSum = 0f;
            for (int neighbourIndex = 0; neighbourIndex < neighbours.Count; neighbourIndex++)
            {
                (Sample trainSample, float distance, float weight) = neighbours[neighbourIndex];
                weightSum += weight;
                for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
                {
                    testPredictions[testIndex, outputIndex] += trainSample.output[outputIndex] * weight;
                }
            }
            for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
            {
                testPredictions[testIndex, outputIndex] /= weightSum;
            }
        });
    }

    public static void ComputeExponential(List<Sample> train, float[,] inverseNormalizedDistances, int testIndex, float exponent, float[,] testPredictions)
    {
        float weightSum = 0f;
        for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
        {
            Sample sample = train[trainIndex];
            float weight = inverseNormalizedDistances[testIndex, trainIndex];
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
        Parallel.For(0, testInputs.Count, testIndex =>
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
        });
    }

    public static void ComputeLinearK(List<Sample> train, List<float[]> testInputs, float[,] testPredictions, int k)
    {
        Parallel.For(0, testInputs.Count, testIndex =>
        {
            float[] testInput = testInputs[testIndex];
            List<(Sample sample, float distance)> distances = new List<(Sample sample, float distance)>(train.Count);
            for (int trainIndex = 0; trainIndex < train.Count; trainIndex++)
            {
                Sample sample = train[trainIndex];
                float distance = Utility.EuclideanDistance(testInput, sample.input);
                distances.Add((sample, distance));
            }

            // sort near to far, take k
            distances = distances.OrderBy(a => a.distance).Take(k).ToList();

            // find max distance
            float maxDistance = distances.Max(a => a.distance);

            // iterate weightedReciprocalDistances
            //float sumWeight = 0f;
            float[] debug = new float[train[0].output.Length];
            foreach ((Sample sample, float distance) in distances)
            {
                //float weight = 1f - (distance / maxDistance);
                //sumWeight += weight;
                for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
                {
                    testPredictions[testIndex, outputIndex] += sample.output[outputIndex];// * weight;
                    debug[outputIndex] += sample.output[outputIndex];
                }
            }
            for (int outputIndex = 0; outputIndex < train[0].output.Length; outputIndex++)
            {
                testPredictions[testIndex, outputIndex] /= k;
                debug[outputIndex] /= k;
            }
        });
    }
}