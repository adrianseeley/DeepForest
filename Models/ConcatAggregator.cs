public class ConcatAggregator : Model
{
    public List<Model> models;
    public bool verbose;

    public ConcatAggregator(List<Model> models, bool verbose = false)
    {
        this.models = models;
        this.verbose = verbose;
    }

    public override void Fit(List<Sample> samples)
    {
        // iterate models
        for (int modelIndex = 0; modelIndex < models.Count; modelIndex++)
        {
            if (verbose)
            {
                Console.WriteLine($"ConcatAggregator Fitting model {modelIndex + 1} of {models.Count}");
            }

            // get model
            Model model = models[modelIndex];

            // fit model
            model.Fit(samples);
        }
    }

    public override float[] Predict(float[] input)
    {
        // predict each model
        List<float[]> predictions = new List<float[]>(models.Count);
        foreach (Model model in models)
        {
            predictions.Add(model.Predict(input));
        }

        // concat predictions
        float[] concatOutput = Utility.Concat(predictions);
        return concatOutput;
    }
}