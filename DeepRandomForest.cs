public class DeepRandomForest
{
    public int yComponent;
    public List<List<RandomForest>> randomForestLayers;

    public DeepRandomForest(List<(List<float> input, List<float> output)> samples, int yComponent, int treesPerForest, int forestsPerLayer, int layers, bool extraRandom)
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
                // create a random forest for this layer
                RandomForest randomForest = new RandomForest(trainingSamples, yComponent, treesPerForest, extraRandom);
                randomForestLayer.Add(randomForest);
            }

            // iterate through all the training samples
            foreach ((List<float> input, List<float> output) trainingSample in trainingSamples)
            {
                // for each forest in the previous layer
                foreach (RandomForest randomForest in randomForestLayer)
                {
                    // add the prediction of the forest to the input of this training sample (concat)
                    trainingSample.input.Add(randomForest.Predict(trainingSample.input));
                }
            }
        }

        // create the last random forest layer (which is only a single forest)
        RandomForest lastRandomForest = new RandomForest(trainingSamples, yComponent, treesPerForest, extraRandom);
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