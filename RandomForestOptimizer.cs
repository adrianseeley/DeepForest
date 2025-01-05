public static class RandomForestOptimizer
{
    public static WeightedRandomForest AdaptiveBoost(List<Sample> samples, int xComponentCount, int treeCount, int minSamplesPerLeaf, float boostingRate, bool verbose)
    {
        if (verbose)
        {
            Console.WriteLine($"Adaptive Boosted Random Forest, Samples: {samples.Count}, X Components: {xComponentCount}, Trees: {treeCount}, Min Samples Per Leaf: {minSamplesPerLeaf}, BoostingRate: {boostingRate}");
        }

        // create a weighted list of samples
        List<WeightedSample> weightedSamples = new List<WeightedSample>(samples.Count);

        // copy in the samples at 1 weight
        foreach (Sample sample in samples)
        {
            weightedSamples.Add(new WeightedSample(sample.input, sample.output, 1f));
        }

        // create the forest with 1 starting tree
        WeightedRandomForest wrf = new WeightedRandomForest(weightedSamples, xComponentCount, 1, minSamplesPerLeaf, verbose);

        // iterate through remaining trees
        while (wrf.weightedRandomTrees.Count < treeCount)
        {
            // calculate average error
            float averageError = 0f;

            // iterate train samples
            foreach (WeightedSample weightedSample in weightedSamples)
            {
                // get prediction
                float[] prediction = wrf.Predict(weightedSample.input);

                // calculate error
                float error = Error.EuclideanDistance(weightedSample.output, prediction);

                // sum average error
                averageError += error;

                // calculate boost amount
                float boostAmount = error * boostingRate;

                // boost
                weightedSample.weight += boostAmount;
            }

            // calculate average error
            averageError /= weightedSamples.Count;

            if (verbose)
            {
                Console.WriteLine($"Boosted Trees: {wrf.weightedRandomTrees.Count + 1}/{treeCount}, Average Error: {averageError}");
            }

            // build a new tree using boosted samples
            wrf.AddTree(weightedSamples);
        }

        // return adaptive boosted weighted random forest
        return wrf;
    }

    public static RandomForest AdaptiveBoostArgmax(List<Sample> samples, int xComponentCount, int treeCount, int minSamplesPerLeaf, bool verbose)
    {
        if (verbose)
        {
            Console.WriteLine($"Adaptive Boosted Argmax Random Forest, Samples: {samples.Count}, X Components: {xComponentCount}, Trees: {treeCount}, Min Samples Per Leaf: {minSamplesPerLeaf}");
        }

        // create the forest with 1 starting tree
        RandomForest rf = new RandomForest(samples, xComponentCount, 1, minSamplesPerLeaf, verbose);

        // create a boosted list of samples
        List<Sample> boostedSamples = new List<Sample>(samples);

        // iterate through remaining trees
        while (rf.randomTrees.Count < treeCount)
        {
            // iterate train samples
            foreach(Sample sample in samples)
            {
                // get prediction
                float[] prediction = rf.Predict(sample.input);

                // if not correct, add a copy to boosted samples
                if (!Error.ArgmaxEquals(sample, prediction))
                {
                    boostedSamples.Add(sample);
                }
            }

            if (verbose)
            {
                Console.WriteLine($"Boosted Samples: {boostedSamples.Count}, Trees: {rf.randomTrees.Count}");
            }

            // build a new tree using boosted samples
            rf.AddTree(boostedSamples);
        }

        // return adaptive boosted random forest
        return rf;
    }
}
