public static class KNNOptimizer
{
    public static (List<int> bestFeatures, float bestError) DeterministicGreedyFeatureAddition(int k, List<Sample> samples, Error.ErrorFunction errorFunction, int threadCount, bool verbose)
    {
        ParallelOptions parallelOptions = new ParallelOptions();
        parallelOptions.MaxDegreeOfParallelism = threadCount;

        // track the best result
        List<int> bestFeatures = new List<int>();
        float bestError = float.PositiveInfinity;

        // get the feature count
        int featureCount = samples[0].input.Length;

        // create locks
        object consoleLock = new object();
        object bestLock = new object();

        // loop until no more improvements
        bool improved = true;
        while (improved)
        {
            improved = false;

            if (verbose)
            {
                Console.WriteLine($"New Pass, Features: [{string.Join(", ", bestFeatures)}], Error: {bestError}");
            }
            
            // if we have added all the features, we are done
            if (bestFeatures.Count == featureCount)
            {
                break;
            }

            int bestFeatureAddition = -1;
            float bestAdditionError = bestError;

            // iterate over features to add
            Parallel.For(0, featureCount, parallelOptions, (featureIndex) =>
            {
                // if this feature is already present skip it
                if (bestFeatures.Contains(featureIndex))
                {
                    if (verbose)
                    {
                        lock (consoleLock)
                        {
                            Console.WriteLine($"Skipping Addition (Already Included): {featureIndex}");
                        }
                    }

                    return;
                }

                if (verbose)
                {
                    lock (consoleLock)
                    {
                        Console.WriteLine($"Trying Feature: {featureIndex}");
                    }
                }

                // we can try to add this feature index, create a features copy
                List<int> trialFeatures = new List<int>(bestFeatures);

                // add the current feature
                trialFeatures.Add(featureIndex);

                // create the model
                KNN knn = new KNN(k, samples, trialFeatures);

                // make predictions
                List<float[]> predictions = new List<float[]>();
                foreach (Sample sample in samples)
                {
                    predictions.Add(knn.Predict(sample.input, excludeZero: true));
                }

                // compute error
                float error = errorFunction(samples, predictions);

                // if this is the best addition so far
                lock (bestLock)
                {
                    if (error < bestAdditionError)
                    {
                        // stash it
                        bestFeatureAddition = featureIndex;
                        bestAdditionError = error;

                        if (verbose)
                        {
                            lock (consoleLock)
                            {
                                Console.WriteLine($"Better Feature: {bestFeatureAddition}, Error: {bestAdditionError}");
                            }
                        }
                    }
                }
            });

            // if we found a better feature addition
            if (bestFeatureAddition != -1)
            {
                // add the best feature
                bestFeatures.Add(bestFeatureAddition);
                bestError = bestAdditionError;
                improved = true;
            }
        }

        if (verbose)
        {
            Console.WriteLine($"Complete Features: [{string.Join(", ", bestFeatures)}], Error: {bestError}");
        }

        // search complete
        return (bestFeatures, bestError);
    }
}
