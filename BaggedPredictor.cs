using System;

public abstract class BaggedPredictor<Model>
{
    public List<Sample> samples;
    public List<Model> models;

    public BaggedPredictor(List<Sample> samples)
    {
        this.samples = samples;
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
        Model model = AddModel(baggedSamples);
        models.Add(model);
    }

    protected abstract Model AddModel(List<Sample> baggedSamples);

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