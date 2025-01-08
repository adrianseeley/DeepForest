public class BootstrapAggregator : Model
{
    public List<Model> models;
    public List<List<int>> modelFeatures;
    public float featurePercentPerBag;
    public bool verbose;

    public BootstrapAggregator(float featurePercentPerBag, List<Model> models, bool verbose = false)
    {
        this.models = models;
        this.modelFeatures = new List<List<int>>();
        this.featurePercentPerBag = featurePercentPerBag;
        if (featurePercentPerBag <= 0 || featurePercentPerBag > 1)
        {
            throw new ArgumentException("featurePercentPerBag must be in the range (0, 1]");
        }
        this.verbose = verbose;
    }

    public override void Fit(List<Sample> samples)
    {
        // create a local random
        Random random = new Random();

        // reset model features
        modelFeatures.Clear();
        
        // determine the actual feature count per bag with a min 1 and max of feature count
        int featuresPerBag = (int)(samples[0].input.Length * featurePercentPerBag);
        featuresPerBag = Math.Max(1, featuresPerBag);
        featuresPerBag = Math.Min(samples[0].input.Length, featuresPerBag);

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
            List<int> baggedFeatures = Enumerable.Range(0, samples[0].input.Length).OrderBy(i => random.Next()).Take(featuresPerBag).ToList();

            // add to model features
            modelFeatures.Add(baggedFeatures);

            // reselect bagged sample features
            baggedSamples = Sample.ReselectFeatures(baggedSamples, baggedFeatures);

            // fit model
            model.Fit(baggedSamples);
        }
    }

    public override float[] Predict(float[] input)
    {
        // store each models predictions
        List<float[]> predictions = new List<float[]>(models.Count);
        
        // iterate models
        for (int modelIndex = 0; modelIndex < models.Count; modelIndex++)
        {
            // get model
            Model model = models[modelIndex];

            // get features
            List<int> features = modelFeatures[modelIndex];

            // reselect input features
            float[] reselectedInput = Sample.ReselectFeatures(input, features);

            // predict model with input features
            predictions.Add(model.Predict(reselectedInput));
        }

        // average predictions
        float[] averageOutput = Utility.Average(predictions);
        return averageOutput;
    }
}