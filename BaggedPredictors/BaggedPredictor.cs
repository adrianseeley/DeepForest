public abstract class BaggedPredictor<Model>
{
    public List<Sample> samples;
    public int? featuresPerBag;
    public List<Model> models;

    public BaggedPredictor(List<Sample> samples, int? featuresPerBag)
    {
        this.samples = samples;
        this.featuresPerBag = featuresPerBag;
        this.models = new List<Model>();
    }

    public void Build(int modelCount)
    {
        for (int modelIndex = 0; modelIndex < modelCount; modelIndex++)
        {
            AddModel();
        }
    }

    public void AddModel()
    {
        Random random = new Random();
        List<Sample> baggedSamples = new List<Sample>();
        for (int i = 0; i < samples.Count; i++)
        {
            baggedSamples.Add(samples[random.Next(samples.Count)]);
        }
        List<int> baggedFeatures;
        if (featuresPerBag == null)
        {
            baggedFeatures = Enumerable.Range(0, samples[0].input.Length).ToList();
        }
        else
        {
            baggedFeatures = Enumerable.Range(0, samples[0].input.Length).OrderBy(i => random.Next()).Take(featuresPerBag.Value).ToList();
        }
        Model model = AddModel(baggedSamples, baggedFeatures);
        models.Add(model);
    }

    protected abstract Model AddModel(List<Sample> baggedSamples, List<int> baggedFeatures);

    protected abstract float[] Predict(Model model, float[] input);

    public float[] Predict(float[] input)
    {
        List<float[]> predictions = new List<float[]>(models.Count);
        foreach (Model model in models)
        {
            predictions.Add(Predict(model, input));
        }
        float[] averageOutput = Utility.Average(predictions);
        return averageOutput;
    }
}