using System.Reflection.Metadata.Ecma335;

public class BootstrapAggregator : Model
{
    public List<Model> models;
    public float featurePercentPerBag;
    public bool verbose;

    public BootstrapAggregator(float featurePercentPerBag, List<Model> models, bool verbose = false)
    {
        this.models = models;
        this.featurePercentPerBag = featurePercentPerBag;
        if (featurePercentPerBag <= 0 || featurePercentPerBag > 1)
        {
            throw new ArgumentException("featurePercentPerBag must be in the range (0, 1]");
        }
        this.verbose = verbose;
    }

    public override void Fit(List<Sample> samples, List<int> features)
    {
        // create a local random
        Random random = new Random();
        
        // determine the actual feature count per bag with a min 1 and max of feature count
        int featuresPerBag = (int)(features.Count * featurePercentPerBag);
        featuresPerBag = Math.Max(1, featuresPerBag);
        featuresPerBag = Math.Min(features.Count, featuresPerBag);

        // iterate models
        for (int modelIndex = 0; modelIndex < models.Count; modelIndex++)
        {
            if (verbose)
            {
                Console.WriteLine($"BootstrapAggregator Fitting model {modelIndex + 1} of {models.Count}");
            }

            // get model
            Model model = models[modelIndex];

            // bootstrap samples with replacement
            List<Sample> baggedSamples = new List<Sample>();
            for (int i = 0; i < samples.Count; i++)
            {
                baggedSamples.Add(samples[random.Next(samples.Count)]);
            }

            // bootstrap features without replacement
            List<int> baggedFeatures = features.OrderBy(i => random.Next()).Take(featuresPerBag).ToList();

            // fit model
            model.Fit(baggedSamples, baggedFeatures);
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

        // average predictions
        float[] averageOutput = Utility.Average(predictions);
        return averageOutput;
    }
}