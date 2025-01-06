public class RandomTree
{
    public float[] output;
    public int? splitXComponent;
    public float? splitXValue;
    public RandomTree? left;
    public RandomTree? right;

    public RandomTree(Random random, List<Sample> samples, int minSamplesPerLeaf, int splitAttempts, float flipRate)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        // create output
        output = new float[samples[0].output.Length];
        foreach(Sample sample in samples)
        {
            for (int i = 0; i < output.Length; i++)
            {
                output[i] += sample.output[i];
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= (float)samples.Count;
        }

        // initialize split criteria to null, so if we find no split, we can return
        this.splitXComponent = null;
        this.splitXValue = null;
        this.left = null;
        this.right = null;

        // if we have only 1 sample, no split can be made
        if (samples.Count == 1)
        {
            return;
        }

        // search for the best split
        List<Sample> leftSamples = new List<Sample>(samples.Count);
        List<Sample> rightSamples = new List<Sample>(samples.Count);

        // try splitting 
        for (int splitAttempt = 0; splitAttempt < splitAttempts; splitAttempt++)
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
                bool flip = false;
                if (random.NextSingle() < flipRate)
                {
                    flip = true;
                }

                if (sample.input[xComponent] <= xValue && !flip)
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
                left = new RandomTree(random, leftSamples, minSamplesPerLeaf, splitAttempts, flipRate);
                right = new RandomTree(random, rightSamples, minSamplesPerLeaf, splitAttempts, flipRate);

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