<!-- this is for ask.php -->
<?php
ob_clean();
header("Content-Type: application/json");
ini_set('display_errors', 1);
error_reporting(E_ALL);

// Read input from frontend
$data = json_decode(file_get_contents("php://input"), true);
$question = $data["question"] ?? "";

// Upload directory
$uploadDir = __DIR__ . "/uploads/";
$files = array_diff(scandir($uploadDir, SCANDIR_SORT_DESCENDING), ['.', '..']);

$pdfFile = '';
foreach ($files as $file) {
    if (strtolower(pathinfo($file, PATHINFO_EXTENSION)) === 'pdf') {
        $pdfFile = $file;
        break;
    }
}

if ($pdfFile && $question) {

    $pdfPath = realpath($uploadDir . '/' . $pdfFile);

    // Prepare data for Flask API
    $postData = json_encode([
        "pdf_path" => $pdfPath,
        "question" => $question
    ]);

    // Call Flask API
    $ch = curl_init("http://127.0.0.1:5000/process");
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, ["Content-Type: application/json"]);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);

    $response = curl_exec($ch);

    if ($response === false) {
        echo json_encode(["answer" => "❌ Flask server not running. Start app.py first."]);
        curl_close($ch);
        exit;
    }

    curl_close($ch);

    echo $response;
    exit;

} else {
    echo json_encode(["answer" => "❌ No PDF found or question missing."]);
    exit;
}



