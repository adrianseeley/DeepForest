public class RandomCairnForest
{
    public Random random;
    public List<Sample> samples;
    public int minSamplesPerLeaf;
    public int maxLeafDepth;
    public RandomCairnTree.Minimize minimize;
    public List<RandomCairnTree> trees;

    public RandomCairnForest(List<Sample> samples, int minSamplesPerLeaf, int maxLeafDepth, RandomCairnTree.Minimize minimize)
    {
        this.random = new Random();
        this.samples = samples;
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.maxLeafDepth = maxLeafDepth;
        this.minimize = minimize;
        this.trees = new List<RandomCairnTree>();
    }

    public void AddTree()
    {
        List<Sample> resamples = new List<Sample>();
        for (int i = 0; i < samples.Count; i++)
        {
            resamples.Add(samples[random.Next(samples.Count)]);
        }
        trees.Add(RandomCairnTree.Build(resamples, minSamplesPerLeaf, maxLeafDepth, minimize));
    }

    public float[] Predict(float[] input)
    {
        float[] output = new float[samples[0].output.Length];
        foreach (RandomCairnTree tree in trees)
        {
            float[] prediction = tree.Predict(input);
            for (int i = 0; i < output.Length; i++)
            {
                output[i] += prediction[i];
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= trees.Count;
        }
        return output;
    }
}