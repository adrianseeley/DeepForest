public static class RandomForestOptimizer
{
    public static RandomForest AdaptiveBoostArgmax(List<Sample> samples, int xComponentCount, int treeCount, int minSamplesPerLeaf, bool verbose)
    {
        if (verbose)
        {
            Console.WriteLine($"Adaptive Boosted Random Forest, Samples: {samples.Count}, X Components: {xComponentCount}, Trees: {treeCount}, Min Samples Per Leaf: {minSamplesPerLeaf}");
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

                // get argmax error
                float error = Error.ArgmaxError(sample, prediction);

                // if the error was 1f (incorrect), add a copy to boosted samples
                if (error == 1f)
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
