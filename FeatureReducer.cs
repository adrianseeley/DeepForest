public static class FeatureReducer
{
    public delegate float ModelTestDelegate(List<Sample> train, List<Sample> validate);

    public static List<int> RemoveHomogenous(List<Sample> samples)
    {
        List<int> features = new List<int>();
        for (int featureIndex = 0; featureIndex < samples[0].input.Length; featureIndex++)
        {
            float firstValue = samples[0].input[featureIndex];
            bool homogenous = true;
            for (int i = 1; i < samples.Count; i++)
            {
                if (samples[i].input[featureIndex] != firstValue)
                {
                    homogenous = false;
                    break;
                }
            }
            if (!homogenous)
            {
                features.Add(featureIndex);
            }
        }
        return features;
    }

    public static List<int> GreedySelectionAdditive(List<Sample> train, List<Sample> validate, ModelTestDelegate modelTestDelegate, int threadCount, bool verbose)
    {
        ParallelOptions parallelOptions = new ParallelOptions();
        parallelOptions.MaxDegreeOfParallelism = threadCount;
        object consoleLock = new object();
        object bestLock = new object();
        int featureCount = train[0].input.Length;
        List<int> bestFeatures = new List<int>();
        float bestError = float.PositiveInfinity;

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

            // track the best feature to add
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

                // build the sample sets
                List<Sample> trainSubset = Sample.ReselectFeatures(train, trialFeatures);
                List<Sample> validateSubset = Sample.ReselectFeatures(validate, trialFeatures);

                // get error from delegate
                float error = modelTestDelegate(trainSubset, validateSubset);

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
        return bestFeatures;
    }

    public static List<int> GreedySelectionSubtractive(List<Sample> train, List<Sample> validate, ModelTestDelegate modelTestDelegate, int threadCount, bool verbose)
    {
        ParallelOptions parallelOptions = new ParallelOptions();
        parallelOptions.MaxDegreeOfParallelism = threadCount;
        object consoleLock = new object();
        object bestLock = new object();
        int featureCount = train[0].input.Length;
        List<int> bestFeatures = Enumerable.Range(0, featureCount).ToList();
        float bestError = modelTestDelegate(train, validate);

        // loop until no more improvements
        bool improved = true;
        while (improved)
        {
            improved = false;

            if (verbose)
            {
                Console.WriteLine($"New Pass, Features: [{string.Join(", ", bestFeatures)}], Error: {bestError}");
            }

            // if we subtracted all but 1 feature, we are done
            if (bestFeatures.Count == 1)
            {
                break;
            }

            // track the best feature to subtract
            int bestFeatureSubtraction = -1;
            float bestSubtractionError = bestError;

            // iterate over features to subtract
            Parallel.For(0, featureCount, parallelOptions, (featureIndex) =>
            {
                // if this feature is already subtracted skip it
                if (!bestFeatures.Contains(featureIndex))
                {
                    if (verbose)
                    {
                        lock (consoleLock)
                        {
                            Console.WriteLine($"Skipping Subtraction (Already Subtracted): {featureIndex}");
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

                // we can try to subtract this feature index, create a features copy
                List<int> trialFeatures = new List<int>(bestFeatures);

                // subtract the current feature
                trialFeatures.Remove(featureIndex);

                // build the sample sets
                List<Sample> trainSubset = Sample.ReselectFeatures(train, trialFeatures);
                List<Sample> validateSubset = Sample.ReselectFeatures(validate, trialFeatures);

                // get error from delegate
                float error = modelTestDelegate(trainSubset, validateSubset);

                // if this is the best subtraction so far
                lock (bestLock)
                {
                    if (error < bestSubtractionError)
                    {
                        // stash it
                        bestFeatureSubtraction = featureIndex;
                        bestSubtractionError = error;

                        if (verbose)
                        {
                            lock (consoleLock)
                            {
                                Console.WriteLine($"Better Feature: {bestFeatureSubtraction}, Error: {bestSubtractionError}");
                            }
                        }
                    }
                }
            });

            // if we found a better feature subtraction
            if (bestFeatureSubtraction != -1)
            {
                // subtract the best feature
                bestFeatures.Remove(bestFeatureSubtraction);
                bestError = bestSubtractionError;
                improved = true;
            }
        }

        if (verbose)
        {
            Console.WriteLine($"Complete Features: [{string.Join(", ", bestFeatures)}], Error: {bestError}");
        }

        // search complete
        return bestFeatures;
    }
}