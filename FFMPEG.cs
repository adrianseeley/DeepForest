using System.Diagnostics;

public class FFMPEG
{
    public static string FFMPEG_PATH = "ffmpeg.exe";
    public enum EncodeAs
    {
        MP4,
        GIF
    };

    public Process process;
    public Stream inputStream;
    public bool finished;

    public FFMPEG(string filename, int framesPerSecond, int width, int height, EncodeAs encodeAs)
    {
        // setup process start info based on encoder choice
        ProcessStartInfo processStartInfo;
        if (encodeAs == EncodeAs.MP4)
        {
            processStartInfo = new ProcessStartInfo
            {
                FileName = FFMPEG_PATH,
                Arguments = $"-y -f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {framesPerSecond} -i pipe:0 -c:v libx264 -pix_fmt yuv420p {filename}",
                RedirectStandardInput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
        }
        else if (encodeAs == EncodeAs.GIF)
        {
            processStartInfo = new ProcessStartInfo
            {
                FileName = FFMPEG_PATH,
                Arguments = $"-y -f rawvideo -pix_fmt bgr24 -s {width} x {height} -r {framesPerSecond} -i pipe:0 -vf \"split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" {filename}",
                RedirectStandardInput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
        }
        else
        {
            throw new Exception("Invalid encodeAs");
        }

        // create the process
        process = new Process { StartInfo = processStartInfo };
        process.Start();

        // capture input stream
        inputStream = process.StandardInput.BaseStream;

        // mark not finished
        finished = false;
    }

    public void AddFrame(byte[,,] pixels)
    {
        // write frame to input stream
        for (int i = 0; i < pixels.GetLength(0); i++)
        {
            for (int j = 0; j < pixels.GetLength(1); j++)
            {
                inputStream.WriteByte(pixels[i, j, 2]);
                inputStream.WriteByte(pixels[i, j, 1]);
                inputStream.WriteByte(pixels[i, j, 0]);
            }
        }
    }

    public void Finish()
    {
        finished = true;
        inputStream.Close();
        process.WaitForExit();
        process.Close();
    }
}
