public class StandardTree
{
    public enum Reduction
    {
        MeanSquaredError,
        MeanAbsoluteError
    };

    public float[] output;
    public float error;
    public int? splitXComponent;
    public float? splitXValue;
    public StandardTree? left;
    public StandardTree? right;

    public StandardTree(List<Sample> samples, List<int> features, int minSamplesPerLeaf, int maxLeafDepth, Reduction reduction, int currentDepth = 0)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        this.output = Sample.AverageOutput(samples);
        this.error = CalculateError(samples, output, reduction);
        this.splitXComponent = null;
        this.splitXValue = null;
        this.left = null;
        this.right = null;

        // if we dont have enough samples for two leaves, we cant split
        if (samples.Count < minSamplesPerLeaf * 2)
        {
            return;
        }

        // if this is the max leaf depth we cant split
        if (maxLeafDepth != -1 && currentDepth >= maxLeafDepth)
        {
            return;
        }

        // search for the best split
        List<Sample> leftSamples = new List<Sample>(samples.Count);
        List<Sample> rightSamples = new List<Sample>(samples.Count);
        float bestSplitError = error;
        int bestSplitXComponent = -1;
        float bestSplitXValue = float.NaN;

        // iterate x components
        foreach(int xIndex in features)
        {
            // get unique x values
            HashSet<float> xValues = Sample.HistogramInputComponent(samples, xIndex);

            // if there is only one unique x value, we cant split on it
            if (xValues.Count == 1)
            {
                continue;
            }

            // iterate x values
            foreach (float xValue in xValues)
            {
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
                if (leftSamples.Count >= minSamplesPerLeaf && rightSamples.Count >= minSamplesPerLeaf)
                {
                    // calculate the error
                    float leftError = CalculateError(leftSamples, Sample.AverageOutput(leftSamples), reduction);
                    float rightError = CalculateError(rightSamples, Sample.AverageOutput(rightSamples), reduction);
                    float leftWeight = (float)leftSamples.Count / samples.Count;
                    float rightWeight = (float)rightSamples.Count / samples.Count;
                    float splitError = (leftWeight * leftError) + (rightWeight * rightError);

                    // if this is the best split so far
                    if (splitError < bestSplitError)
                    {
                        bestSplitError = splitError;
                        bestSplitXComponent = xIndex;
                        bestSplitXValue = xValue;
                    }
                }
            }
        }

        // if we found a split
        if (bestSplitXComponent != -1)
        {
            // mark split
            this.splitXComponent = bestSplitXComponent;
            this.splitXValue = bestSplitXValue;

            // gather left and right
            leftSamples.Clear();
            rightSamples.Clear();
            foreach (Sample sample in samples)
            {
                if (sample.input[bestSplitXComponent] <= bestSplitXValue)
                {
                    leftSamples.Add(sample);
                }
                else
                {
                    rightSamples.Add(sample);
                }
            }

            // create the left and right children
            this.left = new StandardTree(leftSamples, features, minSamplesPerLeaf, maxLeafDepth, reduction, currentDepth + 1);
            this.right = new StandardTree(rightSamples, features, minSamplesPerLeaf, maxLeafDepth, reduction, currentDepth + 1);
        }
    }

    private static float CalculateError(List<Sample> samples, float[] averageOutput, Reduction reduction)
    {
        switch (reduction)
        {
            case Reduction.MeanSquaredError:
                return Sample.OutputMeanSquaredError(samples, averageOutput);
            case Reduction.MeanAbsoluteError:
                return Sample.OutputMeanAbsoluteError(samples, averageOutput);
            default:
                throw new Exception("Unknown reduction.");
        }
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