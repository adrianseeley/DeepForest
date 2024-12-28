public class DeepRandomForest
{
    public int yComponent;
    public List<List<RandomForest>> randomForestLayers;

    public DeepRandomForest(List<(List<float> input, List<float> output)> samples, int yComponent, int treesPerForest, int forestsPerLayer, int layers, bool extraRandom, bool verbose)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        // mark the y component of interest
        this.yComponent = yComponent;

        // initialize the random forest layers list
        this.randomForestLayers = new List<List<RandomForest>>();

        // create a mutation safe list of training samples
        List<(List<float> input, List<float> output)> trainingSamples = new List<(List<float> input, List<float> output)>();
        foreach ((List<float> input, List<float> output) sample in samples)
        {
            // we deep copy the input and output lists to avoid mutating the originals
            List<float> newInput = new List<float>(sample.input);
            List<float> newOutput = new List<float>(sample.output);
            trainingSamples.Add((newInput, newOutput));
        }

        // iterate through all the forest layers (except the last)
        for (int layerIndex = 0; layerIndex < layers - 1; layerIndex++)
        {
            // initialize the random forests list for this layer
            List<RandomForest> randomForestLayer = new List<RandomForest>();

            // add it to the layers
            randomForestLayers.Add(randomForestLayer);

            // iterate through the forests per layer
            for (int forestIndex = 0; forestIndex < forestsPerLayer; forestIndex++)
            {
                if (verbose)
                {
                    Console.WriteLine($"DRF Inner Layer: {layerIndex}/{layers - 2}, Forest: {forestIndex}/{forestsPerLayer - 1}");
                }

                // choose x component count
                int xComponentCount = (int)Math.Floor(((float)forestIndex) / ((float)forestsPerLayer) * ((float)samples[0].input.Count));
                xComponentCount = Math.Min(Math.Max(1, xComponentCount), samples[0].input.Count);

                // create a random forest for this layer
                RandomForest randomForest = new RandomForest(trainingSamples, yComponent, xComponentCount, treesPerForest, extraRandom);
                randomForestLayer.Add(randomForest);
            }

            // iterate through all the training samples
            int fed = 0;
            object fedLock = new object();
            Parallel.For(0, trainingSamples.Count, sampleIndex =>
            {
                int localSampleIndex = sampleIndex;
                (List<float> input, List<float> output) trainingSample = trainingSamples[localSampleIndex];

                // for each forest in the previous layer
                foreach (RandomForest randomForest in randomForestLayer)
                {
                    // add the prediction of the forest to the input of this training sample (concat)
                    trainingSample.input.Add(randomForest.Predict(trainingSample.input));
                }

                if (verbose)
                {
                    lock(fedLock)
                    {
                        Console.Write($"\rDRF Inner Layer Fed: {fed}/{trainingSamples.Count - 1}");
                        fed++;
                    }
                }
            });
            if (verbose)
            {
                Console.WriteLine();
            }
        }

        if (verbose)
        {
            Console.WriteLine($"DRF Outer Layer");
        }

        // create the last random forest layer (which is only a single forest)
        RandomForest lastRandomForest = new RandomForest(trainingSamples, yComponent, samples[0].input.Count, treesPerForest, extraRandom);
        randomForestLayers.Add(new List<RandomForest>() { lastRandomForest });
    }

    public float Predict(List<float> input)
    {
        // create a mutation safe list of the input
        List<float> predictInput = new List<float>(input);

        // iterate through all the forest layers (except the last)
        for (int layerIndex = 0; layerIndex < randomForestLayers.Count - 1; layerIndex++)
        {
            // get the random forest layer at the current index
            List<RandomForest> randomForestLayer = randomForestLayers[layerIndex];

            // iterate through all the forests in the layer
            foreach (RandomForest randomForest in randomForestLayer)
            {
                // add the prediction of the forest to the input (concat)
                predictInput.Add(randomForest.Predict(predictInput));
            }
        }

        // get the last random forest layer
        List<RandomForest> lastRandomForestLayer = randomForestLayers[randomForestLayers.Count - 1];
        
        // get the last random forest (there is only one in the final layer
        RandomForest lastRandomForest = lastRandomForestLayer[0];

        // return the prediction of the last random forest
        return lastRandomForest.Predict(predictInput);
    }
}