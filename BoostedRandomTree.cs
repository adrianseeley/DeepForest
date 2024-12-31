public class BoostedRandomTree
{
    public int outputComponentCount;
    public float learningRate;
    public List<RandomTree> randomTrees;

    public BoostedRandomTree(Random random, List<Sample> samples, List<int> xComponents, int treeCount, float learningRate)
    {
        this.outputComponentCount = samples[0].output.Length;
        this.learningRate = learningRate;
        this.randomTrees = new List<RandomTree>(treeCount);

        List<float[]> currentPredictions = new List<float[]>(samples.Count);
        for (int i = 0; i < samples.Count; i++)
        {
            currentPredictions.Add(new float[samples[i].output.Length]);
        }

        for (int treeIndex = 0; treeIndex < treeCount; treeIndex++)
        {
            List<Sample> residuals = new List<Sample>(samples.Count);
            for (int i = 0; i < samples.Count; i++)
            {
                float[] actual = samples[i].output;
                float[] currentPrediction = currentPredictions[i];
                float[] residual = new float[actual.Length];

                for (int j = 0; j < actual.Length; j++)
                {
                    residual[j] = actual[j] - currentPrediction[j];
                }
                residuals.Add(new Sample(samples[i].input, residual));
            }

            RandomTree randomTree = new RandomTree(random, residuals, xComponents);
            randomTrees.Add(randomTree);

            for (int i = 0; i < samples.Count; i++)
            {
                float[] prediction = randomTree.Predict(samples[i].input);
                for (int j = 0; j < prediction.Length; j++)
                {
                    currentPredictions[i][j] += learningRate * prediction[j];
                }
            }
        }
    }

    public float[] Predict(float[] input)
    {
        float[] sum = new float[outputComponentCount];
        foreach (RandomTree randomTree in randomTrees)
        {
            float[] prediction = randomTree.Predict(input);
            for (int j = 0; j < outputComponentCount; j++)
            {
                sum[j] += learningRate * prediction[j];
            }
        }
        return sum;
    }
}