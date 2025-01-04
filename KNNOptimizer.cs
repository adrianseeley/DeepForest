public static class KNNOptimizer
{
    public static (List<int> bestFeatures, float bestError) DeterministicGreedyFeatureAddition(int k, List<Sample> train, List<Sample> validation, Error.ErrorFunction errorFunction, bool verbose)
    {
        // track the best result
        List<int> bestFeatures = new List<int>();
        float bestError = float.PositiveInfinity;

        // get the feature count
        int featureCount = train[0].input.Length;

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

            // iterate over features to add
            for (int featureIndex = 0; featureIndex < featureCount; featureIndex++)
            {
                // if this feature is already present skip it
                if (bestFeatures.Contains(featureIndex))
                {
                    continue;
                }

                // we can try to add this feature index, create a features copy
                List<int> trialFeatures = new List<int>(bestFeatures);

                // add the current feature
                trialFeatures.Add(featureIndex);

                // create the model
                KNN knn = new KNN(k, train, trialFeatures);

                // make predictions
                List<float[]> predictions = new List<float[]>();
                foreach (Sample sample in validation)
                {
                    predictions.Add(knn.Predict(sample.input));
                }

                // compute error
                float error = errorFunction(validation, predictions);

                // if this is the best error, update the best features
                if (error < bestError)
                {
                    bestError = error;
                    bestFeatures = trialFeatures;
                    improved = true;

                    if (verbose)
                    {
                        Console.WriteLine($"New Best, Features: [{string.Join(", ", bestFeatures)}], Error: {bestError}");
                    }
                }
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
