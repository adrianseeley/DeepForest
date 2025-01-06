/// <summary>
/// Samples are assumed to be prenormalized in both the input and output dimensions.
/// </summary>

public class GraphDiffusion
{
    public static int debugWidth = 1280;
    public static int debugHeight = 1280;
    public static int debugK = 10;
    public static bool pixelNeighboursCalculated = false;
    public static List<GraphDiffusionNode>[,] pixelNeighbours = new List<GraphDiffusionNode>[debugWidth, debugHeight];
    public static FFMPEG ffmpeg = new FFMPEG("./result.mp4", 20, debugWidth, debugHeight, FFMPEG.EncodeAs.MP4);


    public List<GraphDiffusionNode> nodes;

    public GraphDiffusion(List<Sample> samples, int k, int diffusionSteps)
    {
        // convert all samples to nodes
        nodes = new List<GraphDiffusionNode>();
        foreach (Sample sample in samples)
        {
            nodes.Add(new GraphDiffusionNode(sample));
        }

        // we can add non-producing nodes here if needed
        Random random = new Random();
        for (int i = 0; i < 10000; i++)
        {
            float rx = random.NextSingle();
            float ry = random.NextSingle();
            nodes.Add(new GraphDiffusionNode(new float[] { rx, ry }, nodes[0].output.Length));
        }

        // debug stuff
        CalculatePixelNeighbours();
        DebugDrawSamples();
        DebugDrawNonDiffused();

        // calculate distances between all nodes (we skip a==b which defaults to 0 anyways)
        float[,] distances = new float[nodes.Count, nodes.Count];
        float distanceMin = float.PositiveInfinity;
        float distanceMax = float.NegativeInfinity;
        for (int a = 0; a < nodes.Count; a++)
        {
            for (int b = a + 1; b < nodes.Count; b++)
            {
                float distance = EuclideanDistance(nodes[a].input, nodes[b].input);
                distances[a, b] = distance;
                distances[b, a] = distance;
                distanceMin = MathF.Min(distanceMin, distance);
                distanceMax = MathF.Max(distanceMax, distance);
            }
        }
        float distanceRange = distanceMax - distanceMin;

        // calculate normalized distances between all nodes (again skipping a==b which defaults to 0)
        float[,] normalizedDistances = new float[nodes.Count, nodes.Count];
        for (int a = 0; a < nodes.Count; a++)
        {
            for (int b = a + 1; b < nodes.Count; b++)
            {
                float normalizedDistance = (distances[a, b] - distanceMin) / distanceRange;
                normalizedDistances[a, b] = normalizedDistance;
                normalizedDistances[b, a] = normalizedDistance;
            }
        }

        // each node requires a list of neighbours to pull from, find those neighbours
        for (int nodeIndex = 0; nodeIndex < nodes.Count; nodeIndex++)
        {
            // get the node in question
            GraphDiffusionNode node = nodes[nodeIndex];
            
            // create a list of neighbour distances
            List<(GraphDiffusionNode otherNode, float distance)> neighbourDistances = new List<(GraphDiffusionNode otherNode, float distance)>();

            // iterate through all other nodes to find the neighbours
            for (int otherNodexIndex = 0; otherNodexIndex < nodes.Count; otherNodexIndex++)
            {
                // cant have self as a neighbour
                if (otherNodexIndex == nodeIndex)
                {
                    continue;
                }

                // add to list
                neighbourDistances.Add((nodes[otherNodexIndex], normalizedDistances[nodeIndex, otherNodexIndex]));
            }

            // sort by distance close to far
            neighbourDistances.Sort((a, b) => a.distance.CompareTo(b.distance));

            // take the k nearest neighbours
            node.neighbours = neighbourDistances.Take(k).Select(a => a.otherNode).ToList();
        }

        // create targets
        List<float[]> nodeTargets = new List<float[]>();
        foreach (GraphDiffusionNode node in nodes)
        {
            nodeTargets.Add(new float[node.output.Length]);
        }

        // make diffusion steps
        for (int diffusionStep = 0; diffusionStep < diffusionSteps; diffusionStep++)
        {
            Console.WriteLine($"diffusion step: {diffusionStep}/{diffusionSteps}");

            // debug draw
            DebugDraw(diffusionStep);

            // iterate through nodes to generate the targets
            for (int nodeIndex = 0; nodeIndex < nodes.Count; nodeIndex++)
            {
                // get the node
                GraphDiffusionNode node = nodes[nodeIndex];

                // get the node target
                float[] nodeTarget = nodeTargets[nodeIndex];

                // clear it
                Array.Clear(nodeTarget, 0, nodeTarget.Length);

                // track the influence weight sum
                float influenceWeightSum = 0f;

                // if this node is a producer add its influence
                if (node.sample != null)
                {
                    for (int i = 0; i < nodeTarget.Length; i++)
                    {
                        nodeTarget[i] += node.sample.output[i];
                    }
                    influenceWeightSum += 1f;
                }

                // iterate through the neighbours to influence the target
                foreach(GraphDiffusionNode neighbour in node.neighbours)
                {
                    // add influence
                    for (int i = 0; i < nodeTarget.Length; i++)
                    {
                        nodeTarget[i] += neighbour.output[i];
                    }
                    influenceWeightSum += 1f;
                }

                // weighted average the target by the influence weight sum
                for (int i = 0; i < nodeTarget.Length; i++)
                {
                    nodeTarget[i] /= influenceWeightSum;
                }
            }

            // iterate through nodes to update the outputs based on targets
            for (int nodeIndex = 0; nodeIndex < nodes.Count; nodeIndex++)
            {
                // get the node
                GraphDiffusionNode node = nodes[nodeIndex];

                // get the node target
                float[] nodeTarget = nodeTargets[nodeIndex];

                // update the node output (average it with the target)
                for (int i = 0; i < node.output.Length; i++)
                {
                    node.output[i] += nodeTarget[i];
                    node.output[i] /= 2f;
                }
            }
        }

        ffmpeg.Finish();
    }

    public void CalculatePixelNeighbours()
    {
        Console.WriteLine("calculating pixel neighbours, debug...");
        for (int x = 0; x < debugWidth; x++)
        {
            Console.Write($"\r{x}/{debugWidth}");
            List<List<GraphDiffusionNode>> pixelNeighboursChild = new List<List<GraphDiffusionNode>>();
            Parallel.For(0, debugHeight, y =>
            {
                float nx = (float)x / (float)debugWidth;
                float ny = (float)y / (float)debugHeight;
                List<(GraphDiffusionNode node, float distance)> nodeDistances = new List<(GraphDiffusionNode node, float distance)>();
                foreach (GraphDiffusionNode node in nodes)
                {
                    float distance = EuclideanDistance(new float[] { nx, ny }, node.input);
                    nodeDistances.Add((node, distance));
                }
                nodeDistances.Sort((a, b) => a.distance.CompareTo(b.distance));
                pixelNeighbours[x, y] = nodeDistances.Take(debugK).Select(i => i.node).ToList();
            });
        }
        Console.WriteLine();
    }

    public void DebugDrawSamples()
    {
        int width = debugWidth;
        int height = debugHeight;
        byte[,,] bitmap = Bitmap.Create(width, height);

        // draw samples
        foreach (GraphDiffusionNode node in nodes)
        {
            if (node.sample == null)
            {
                continue;
            }
            int x = (int)(node.input[0] * width);
            int y = (int)(node.input[1] * height);
            byte r = (byte)(node.sample.output[0] * 255);
            byte b = (byte)(node.sample.output[1] * 255);
            Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
        }

        Bitmap.SaveToBMP($"./bmp/samples.bmp", bitmap);
    }

    public void DebugDrawNonDiffused()
    {
        int k = debugK;
        int width = debugWidth;
        int height = debugHeight;
        byte[,,] bitmap = Bitmap.Create(width, height);

        // draw predictions
        for (int x = 0; x < width; x++)
        {
            Console.Write($"\rfixed x: {x}/{width}");
            Parallel.For(0, height, y =>
            {
                float nx = (float)x / (float)width;
                float ny = (float)y / (float)height;
                float[] prediction = LookupKAverageNonDiffused(new float[] { nx, ny }, k);
                byte r = (byte)(prediction[0] * 255);
                byte b = (byte)(prediction[1] * 255);
                Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
            });
        }

        // draw samples
        foreach (GraphDiffusionNode node in nodes)
        {
            if (node.sample == null)
            {
                continue;
            }
            int x = (int)(node.input[0] * width);
            int y = (int)(node.input[1] * height);
            byte r = (byte)(node.sample.output[0] * 255);
            byte b = (byte)(node.sample.output[1] * 255);
            Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
        }

        Bitmap.SaveToBMP($"./bmp/nondiffused.bmp", bitmap);
        Console.WriteLine();
    }

    public void DebugDraw(int diffusionStep)
    {
        int k = debugK;
        int width = debugWidth;
        int height = debugHeight;
        byte[,,] bitmap = Bitmap.Create(width, height);
        
        // draw predictions
        for (int x = 0; x < width; x++)
        {
            Parallel.For(0, height, y =>
            {
                float[] prediction = AverageNodes(pixelNeighbours[x, y]);
                byte r = (byte)(prediction[0] * 255);
                byte b = (byte)(prediction[1] * 255);
                Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
            });
        }

        // draw samples
        foreach(GraphDiffusionNode node in nodes)
        {
            if (node.sample == null)
            {
                continue;
            }
            int x = (int)(node.input[0] * width);
            int y = (int)(node.input[1] * height);
            byte r = (byte)(node.sample.output[0] * 255);
            byte b = (byte)(node.sample.output[1] * 255);
            Bitmap.DrawPixel(bitmap, x, y, r, 0, b);
        }

        ffmpeg.AddFrame(bitmap);
    }

    public float[] AverageNodes(List<GraphDiffusionNode> nodes)
    {
        float[] average = new float[nodes[0].output.Length];
        foreach (GraphDiffusionNode node in nodes)
        {
            for (int i = 0; i < average.Length; i++)
            {
                average[i] += node.output[i];
            }
        }
        for (int i = 0; i < average.Length; i++)
        {
            average[i] /= nodes.Count;
        }
        return average;
    }

    public float[] LookupKAverageNonDiffused(float[] input, int k)
    {
        List<(GraphDiffusionNode node, float distance)> nodeDistances = new List<(GraphDiffusionNode node, float distance)>();
        foreach (GraphDiffusionNode node in nodes)
        {
            if (node.sample == null)
            {
                continue;
            }
            float distance = EuclideanDistance(input, node.input);
            nodeDistances.Add((node, distance));
        }
        nodeDistances.Sort((a, b) => a.distance.CompareTo(b.distance));
        int minK = Math.Min(k, nodeDistances.Count);
        float[] output = new float[nodes[0].output.Length];
        for (int i = 0; i < minK; i++)
        {
            GraphDiffusionNode node = nodeDistances[i].node;
            if (node.sample == null)
            {
                throw new Exception("Unreachable.");
            }
            for (int j = 0; j < output.Length; j++)
            {
                output[j] += node.sample.output[j];
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= minK;
        }
        return output;
    }

    public float[] LookupKAverage(float[] input, int k)
    {
        List<(GraphDiffusionNode node, float distance)> nodeDistances = new List<(GraphDiffusionNode node, float distance)>();
        foreach (GraphDiffusionNode node in nodes)
        {
            float distance = EuclideanDistance(input, node.input);
            nodeDistances.Add((node, distance));
        }
        nodeDistances.Sort((a, b) => a.distance.CompareTo(b.distance));
        int minK = Math.Min(k, nodeDistances.Count);
        float[] output = new float[nodes[0].output.Length];
        for (int i = 0; i < minK; i++)
        {
            GraphDiffusionNode node = nodeDistances[i].node;
            for (int j = 0; j < output.Length; j++)
            {
                output[j] += node.output[j];
            }
        }
        for (int i = 0; i < output.Length; i++)
        {
            output[i] /= minK;
        }
        return output;
    }

    public static float EuclideanDistance(float[] a, float[] b)
    {
        float distance = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            distance += MathF.Pow(a[i] - b[i], 2);
        }
        return MathF.Sqrt(distance);
    }
}

public class GraphDiffusionNode
{
    public float[] input;
    public float[] output;
    public Sample? sample;
    public List<GraphDiffusionNode> neighbours;

    public GraphDiffusionNode(float[] input, int outputSize)
    {
        this.input = input;
        this.output = new float[outputSize];
        this.sample = null;
        this.neighbours = new List<GraphDiffusionNode>();
    }

    public GraphDiffusionNode(Sample sample)
    {
        this.input = new float[sample.input.Length];
        sample.input.CopyTo(this.input, 0);
        this.output = new float[sample.output.Length];
        this.sample = sample;
        this.neighbours = new List<GraphDiffusionNode>();
    }
}