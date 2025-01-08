public class RandomTree : Model
{
    public int minSamplesPerLeaf;
    public int? maxLeafDepth;
    public int maxSplitAttempts;
    public int currentDepth;
    public float[]? output;
    public int? splitXComponent;
    public float? splitXValue;
    public RandomTree? left;
    public RandomTree? right;

    public RandomTree(int minSamplesPerLeaf, int? maxLeafDepth, int maxSplitAttempts, int currentDepth = 0)
    {
        this.minSamplesPerLeaf = minSamplesPerLeaf;
        this.maxLeafDepth = maxLeafDepth;
        this.maxSplitAttempts = maxSplitAttempts;
        this.currentDepth = currentDepth;
        this.output = null;
        this.splitXComponent = null;
        this.splitXValue = null;
        this.left = null;
        this.right = null;
    }

    public override void Fit(List<Sample> samples)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        // create output
        output = Sample.AverageOutput(samples);
        this.splitXComponent = null;
        this.splitXValue = null;
        this.left = null;
        this.right = null;

        // if we dont have enough samples for two leaves we cant split
        if (samples.Count < minSamplesPerLeaf * 2)
        {
            return;
        }

        // if this is the max leaf depth we cant split
        if (maxLeafDepth != null && currentDepth >= maxLeafDepth)
        {
            return;
        }

        // search for the best split
        Random random = new Random();
        List<Sample> leftSamples = new List<Sample>(samples.Count);
        List<Sample> rightSamples = new List<Sample>(samples.Count);

        // try splitting 
        for (int splitAttempt = 0; splitAttempt < maxSplitAttempts; splitAttempt++)
        {
            // choose a random x index
            int xIndex = random.Next(samples[0].input.Length);

            // choose a random x value
            float xValue = samples[random.Next(samples.Count)].input[xIndex];

            // gather the left and right samples
            leftSamples.Clear();
            rightSamples.Clear();
            foreach (Sample sample in samples)
            {
                if (sample.input[xIndex] <= xValue)
                {
                    leftSamples.Add(sample);
                }
                else
                {
                    rightSamples.Add(sample);
                }
            }

            // if the split is valid
            if (leftSamples.Count > minSamplesPerLeaf && rightSamples.Count > minSamplesPerLeaf)
            {
                // mark split
                splitXComponent = xIndex;
                splitXValue = xValue;

                // create the left and right children
                left = new RandomTree(minSamplesPerLeaf, maxLeafDepth, maxSplitAttempts, currentDepth + 1);
                right = new RandomTree(minSamplesPerLeaf, maxLeafDepth, maxSplitAttempts, currentDepth + 1);

                // fit
                left.Fit(leftSamples);
                right.Fit(rightSamples);

                // done
                return;
            }
        }

        // reaching here means no split was found
    }

    public override float[] Predict(float[] input)
    {
        // if this is a leaf node, return the y value
        if (splitXComponent == null)
        {
            if (output == null)
            {
                throw new ArgumentNullException("output");
            }
            return output;
        }

        // if the sample goes to the left
        if (input[splitXComponent.Value] <= splitXValue)
        {
            // safety check left child
            if (left == null)
            {
                throw new InvalidOperationException("Left child must not be null.");
            }

            // return the left child prediction
            return left.Predict(input);
        }
        // otherwise the sample goes to the right
        else
        {
            // safety check right child
            if (right == null)
            {
                throw new InvalidOperationException("Right child must not be null.");
            }

            // return the right child prediction
            return right.Predict(input);
        }
    }
}