import { useEffect, useMemo, useState } from "react";

const IDX_TO_NAME = [
  "Hip breadth",
  "Right hip flexion",
  "Left hip flexion",
  "Right hip joint",
  "Right knee flexion",
  "Left hip joint angle",
  "Left knee flexion",
  "Spinal extension",
  "Lower cervical angle",
  "Left neck-to-shoulder angle",
  "Right neck-to-shoulder angle",
  "Upper left shoulder angle",
  "Upper right shoulder angle",
  "Shoulder spread",
  "Head tilt angle",
  "Left shoulder abduction",
  "Left elbow flexion",
  "Right shoulder abduction",
  "Right elbow flexion",
];

export function VideoUploadForm(props: { apiUrl: string; reference: any }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>("");
  const [messageType, setMessageType] = useState<
    "success" | "error" | "info" | ""
  >("");
  const [responseData, setResponseData] = useState<any | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);

  const parsedResponseData = useMemo(() => {
    if (!responseData) return [];
    let ret = [];
    for (let i = 0; i < 19; ++i) {
      ret.push(responseData.joint_loss[`${i}`]);
    }

    ret.sort((a, b) => b - a);

    return ret;
  }, [responseData]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setMessage(""); // Clear previous messages
      setMessageType("");
      setResponseData(null); // Clear previous response data
    } else {
      setSelectedFile(null);
    }
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!selectedFile) {
      setMessage("Please select a video file.");
      setMessageType("error");
      return;
    }

    setUploading(true);
    setMessage("Uploading...");
    setMessageType("info");
    setResponseData(null); // Clear previous response data

    const formData = new FormData();
    formData.append("file", selectedFile);

    // Adding the 'weights' object as in your original script
    const weightsObj: { [key: number]: number } = {};
    for (let i = 0; i < 19; ++i) {
      weightsObj[i] = 1;
    }
    formData.append("weights", JSON.stringify(weightsObj));

    try {
      const response = await fetch(props.apiUrl, {
        method: "POST",
        body: formData,
        // Headers are generally not needed for FormData with fetch,
        // as the browser sets 'Content-Type': 'multipart/form-data' automatically.
      });

      const data: any = await response.json(); // Assuming the server responds with JSON

      if (response.ok) {
        setMessage(`Upload successful!`);
        setMessageType("success");
        setResponseData(data); // Store the response data for rendering
        // Optionally reset the file input, though it's often better to let the user see what they uploaded
        // setSelectedFile(null);
        // (event.target as HTMLFormElement).reset(); // This can also work

        const video_res = await fetch(
          `http://localhost:5000/videos/${data.video_name}`,
        );
        const videoBlob = await video_res.blob();
        setVideoUrl(URL.createObjectURL(videoBlob));
      } else {
        setMessage(
          `Upload failed: ${(data as any).message || response.statusText}`,
        );
        setMessageType("error");
        setResponseData(null);
      }
    } catch (error: any) {
      console.error("Error uploading file:", error);
      setMessage(`An error occurred: ${error.message}`);
      setMessageType("error");
      setResponseData(null);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <section className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-2xl font-semibold text-gray-700 mb-4">
          Upload Your Video
        </h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="videoFile"
              className="block text-sm font-medium text-gray-700 mb-1">
              Select video file:
            </label>
            <input
              type="file"
              id="videoFile"
              name="videoFile"
              accept="video/*"
              required
              onChange={handleFileChange}
              disabled={uploading}
              className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>
          <button
            type="submit"
            disabled={uploading || !selectedFile}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg focus:outline-none focus:shadow-outline transition duration-150 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed">
            {uploading ? "Uploading..." : "Upload Video"}
          </button>
        </form>
        {message && (
          <div
            className={`mt-4 text-sm ${
              messageType === "success"
                ? "text-green-600"
                : messageType === "error"
                ? "text-red-600"
                : "text-blue-600" // info
            }`}>
            {message}
          </div>
        )}

        <p className="text-red-500 hidden"></p>
        <p className="text-orange-500 hidden"></p>
        <p className="text-yellow-500 hidden"></p>
        <p className="text-green-500 hidden"></p>

        {parsedResponseData ? (
          <div className="pt-12 flex flex-row justify-between">
            <div className="shrink-0">
              {parsedResponseData.map((loss: number, idx: number) => (
                <p
                  key={idx}
                  className={
                    loss > 15
                      ? "text-red-500"
                      : loss > 10
                      ? "text-orange-500"
                      : loss > 5
                      ? "text-yellow-500"
                      : "text-green-500"
                  }>
                  {IDX_TO_NAME[idx]}
                </p>
              ))}
            </div>

            {videoUrl ? (
              <video
                className="w-full h-full object-contain rounded-md"
                controls
                autoPlay>
                <source src={videoUrl} type="video/mp4" />
              </video>
            ) : null}
          </div>
        ) : null}
      </section>
    </div>
  );
}
