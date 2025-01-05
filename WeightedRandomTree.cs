public class WeightedRandomTree
{
    public float[] output;
    public int? splitXComponent;
    public float? splitXValue;
    public WeightedRandomTree? left;
    public WeightedRandomTree? right;

    public WeightedRandomTree(Random random, List<WeightedSample> weightedSamples, List<int> xComponents, int minSamplesPerLeaf)
    {
        // confirm we have weightedSamples to work with
        if (weightedSamples.Count == 0)
        {
            throw new ArgumentException("WeightedSamples must not be empty.", nameof(weightedSamples));
        }

        // create output
        output = new float[weightedSamples[0].output.Length];
        float weight = 0f;
        foreach(WeightedSample weightedSample in weightedSamples)
        {
            weight += weightedSample.weight;
            for (int i = 0; i < output.Length; i++)
            {
                output[i] += weightedSample.output[i] * weightedSample.weight;
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= weight;
        }

        // initialize split criteria to null, so if we find no split, we can return
        this.splitXComponent = null;
        this.splitXValue = null;
        this.left = null;
        this.right = null;

        // if we have only 1 weightedSample, no split can be made
        if (weightedSamples.Count == 1)
        {
            return;
        }

        // search for the best split
        List<WeightedSample> leftWeightedSamples = new List<WeightedSample>(weightedSamples.Count);
        List<WeightedSample> rightWeightedSamples = new List<WeightedSample>(weightedSamples.Count);

        // iterate through the available x components
        foreach (int xComponent in xComponents)
        {
            // create a distinct list of all the x values for the current x component, and shuffle them
            List<float> xValues = weightedSamples.Select(s => s.input[xComponent]).Distinct().OrderBy(x => random.Next()).ToList();

            // if there is only 1 x value we cant split it
            if (xValues.Count == 1)
            {
                continue;
            }

            // iterate through the x values
            foreach (float xValue in xValues)
            {
                // gather the left and right weightedSamples
                leftWeightedSamples.Clear();
                rightWeightedSamples.Clear();
                foreach (WeightedSample weightedSample in weightedSamples)
                {
                    if (weightedSample.input[xComponent] <= xValue)
                    {
                        leftWeightedSamples.Add(weightedSample);
                    }
                    else
                    {
                        rightWeightedSamples.Add(weightedSample);
                    }
                }

                // if the split is valid
                if (leftWeightedSamples.Count > minSamplesPerLeaf && rightWeightedSamples.Count > minSamplesPerLeaf)
                {
                    // mark split
                    splitXComponent = xComponent;
                    splitXValue = xValue;

                    // create the left and right children
                    left = new WeightedRandomTree(random, leftWeightedSamples, xComponents, minSamplesPerLeaf);
                    right = new WeightedRandomTree(random, rightWeightedSamples, xComponents, minSamplesPerLeaf);

                    // done
                    return;
                }
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

        // if the weightedSample goes to the left
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
        // otherwise the weightedSample goes to the right
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