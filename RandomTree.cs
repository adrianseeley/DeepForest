public class RandomTree
{
    public int yComponent;
    public float y;
    public float error;
    public int? splitXComponent;
    public float? splitXValue;
    public RandomTree? left;
    public RandomTree? right;

    public RandomTree(Random random, List<(List<float> input, List<float> output)> samples, List<int> xComponents, int yComponent, bool extraRandom)
    {
        // confirm we have samples to work with
        if (samples.Count == 0)
        {
            throw new ArgumentException("Samples must not be empty.", nameof(samples));
        }

        // mark the y component of interest
        this.yComponent = yComponent;

        // set the y value of the tree to the average of the y component of the samples
        this.y = samples.Select(s => s.output[yComponent]).Average();

        // use the mean absolute error as the error of the tree
        this.error = samples.Select(s => Math.Abs(s.output[yComponent] - y)).Sum() / samples.Count;

        // initialize split criteria to null, so if we find no split, we can return
        this.splitXComponent = null;
        this.splitXValue = null;
        this.left = null;
        this.right = null;

        // if there is no error, we have a homogoneous y component, no split required
        if (error == 0f)
        {
            return;
        }

        // search for the best split
        float bestError = error;
        int bestXComponent = -1;
        float bestXValue = float.NaN;
        List<(List<float> input, List<float> output)> leftSamples = new List<(List<float> input, List<float> output)>(samples.Count);
        List<(List<float> input, List<float> output)> rightSamples = new List<(List<float> input, List<float> output)>(samples.Count);

        // setup a flag for extra random found
        bool extraRandomFound = false;

        // iterate through the available x components
        foreach (int xComponent in xComponents)
        {
            // create a distinct list of all the x values for the current x component, and shuffle them
            List<float> xValues = samples.Select(s => s.input[xComponent]).Distinct().OrderBy(x => random.Next()).ToList();

            // iterate through the x values
            foreach (float xValue in xValues)
            {
                // gather the left and right samples
                leftSamples.Clear();
                rightSamples.Clear();
                foreach ((List<float> input, List<float> output) sample in samples)
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

                // if either the left or right samples are empty, this is not a valid split
                if (leftSamples.Count == 0 || rightSamples.Count == 0)
                {
                    continue;
                }

                // if this is an extra random tree, any valid split is a good split
                if (extraRandom)
                {
                    bestXComponent = xComponent;
                    bestXValue = xValue;
                    extraRandomFound = true;
                    break;
                }

                // calculate the left and right y values
                float leftY = leftSamples.Select(s => s.output[yComponent]).Average();
                float rightY = rightSamples.Select(s => s.output[yComponent]).Average();

                // calculate the left and right errors
                float leftError = leftSamples.Select(s => Math.Abs(s.output[yComponent] - leftY)).Sum() / leftSamples.Count;
                float rightError = rightSamples.Select(s => Math.Abs(s.output[yComponent] - rightY)).Sum() / rightSamples.Count;

                // calculate the weighted joint error of the left and right errors
                float jointError = (leftError * (leftSamples.Count / samples.Count)) + (rightError * (rightSamples.Count / samples.Count));

                // if the joint error is better than the initial error
                if (jointError < bestError)
                {
                    // update the best error, x component, and x value
                    bestError = jointError;
                    bestXComponent = xComponent;
                    bestXValue = xValue;
                }
            }

            // if we found a split for extra random, break
            if (extraRandomFound)
            {
                break;
            }
        }

        // if we found no split better than our starting error, return
        if (bestXComponent == -1)
        {
            return;
        }

        // mark the best split
        splitXComponent = bestXComponent;
        splitXValue = bestXValue;

        // non extra random trees require regathering (extra random is already gathered at this point)
        if (!extraRandom)
        {
            // gather the left and right samples
            leftSamples.Clear();
            rightSamples.Clear();
            foreach ((List<float> input, List<float> output) sample in samples)
            {
                if (sample.input[bestXComponent] <= bestXValue)
                {
                    leftSamples.Add(sample);
                }
                else
                {
                    rightSamples.Add(sample);
                }
            }
        }

        // create the left and right children
        left = new RandomTree(random, leftSamples, xComponents, yComponent, extraRandom);
        right = new RandomTree(random, rightSamples, xComponents, yComponent, extraRandom);
    }

    public float Predict(List<float> input)
    {
        // if this is a leaf node, return the y value
        if (splitXComponent == null)
        {
            return y;
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