public class RandomTree
{
    public float[] output;
    public int? splitXComponent;
    public float? splitXValue;
    public RandomTree? left;
    public RandomTree? right;

    public RandomTree(List<Sample> samples, int minSamplesPerLeaf, int maxLeafDepth, int maxSplitAttempts, int currentDepth = 0)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        // create output
        output = Sample.AverageOutput(samples);

        // initialize split criteria to null, so if we find no split, we can return
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
        if (currentDepth >= maxLeafDepth)
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
            // choose a random x component
            int xComponent = random.Next(samples[0].input.Length);

            // choose a random x value
            float xValue = samples[random.Next(samples.Count)].input[xComponent];

            // gather the left and right samples
            leftSamples.Clear();
            rightSamples.Clear();
            foreach (Sample sample in samples)
            {
                if (sample.input[xComponent] <= xValue)
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
                splitXComponent = xComponent;
                splitXValue = xValue;

                // create the left and right children
                left = new RandomTree(leftSamples, minSamplesPerLeaf, maxLeafDepth, maxSplitAttempts, currentDepth + 1);
                right = new RandomTree(rightSamples, minSamplesPerLeaf, maxLeafDepth, maxSplitAttempts, currentDepth + 1);

                // done
                return;
            }
        }

        // reaching here means no split was found
    }

    public float[] Predict(float[] input)
    {
        // if this is a leaf node, return the y value
        if (splitXComponent == null)
        {
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

    public void CountNodes(ref long count)
    {
        count++;
        if (left != null)
        {
            left.CountNodes(ref count);
        }
        if (right != null)
        {
            right.CountNodes(ref count);
        }
    }
}