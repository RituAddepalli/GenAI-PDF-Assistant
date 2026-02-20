<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    if (isset($_FILES["file"])) {
        $targetDir = "uploads/";
        $fileName = basename($_FILES["file"]["name"]);
        $targetFilePath = $targetDir . $fileName;
        $fileType = strtolower(pathinfo($targetFilePath, PATHINFO_EXTENSION));

        // Only allow PDF
        $allowedTypes = array("pdf");
        if (!in_array($fileType, $allowedTypes)) {
            echo json_encode(["status" => "error", "message" => "Only PDF files are allowed."]);
            exit;
        }

        // Delete all existing PDFs in uploads folder
        foreach (glob($targetDir . "*.pdf") as $oldFile) {
            unlink($oldFile); // Delete each old PDF
        }

        // Create uploads directory if it doesn't exist
        if (!is_dir($targetDir)) {
            mkdir($targetDir, 0777, true);
        }

        // Check for upload error
        if ($_FILES["file"]["error"] !== 0) {
            echo json_encode(["status" => "error", "message" => "Upload error: " . $_FILES["file"]["error"]]);
            exit;
        }

        // Save the new uploaded file
        if (move_uploaded_file($_FILES["file"]["tmp_name"], $targetFilePath)) {
            echo json_encode(["status" => "success", "filename" => $fileName]);
        } else {
            echo json_encode(["status" => "error", "message" => "Failed to move uploaded file."]);
        }
    } else {
        echo json_encode(["status" => "error", "message" => "No file uploaded."]);
    }
} else {
    echo json_encode(["status" => "error", "message" => "Invalid request."]);
}
?>
