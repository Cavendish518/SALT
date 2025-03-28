#include "ProgressDialog.h"
#include <QVBoxLayout>
#include <QPixmap>
#include <QProcess>
#include <QDebug>
#include <QByteArray>

ProgressDialog::ProgressDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle("SALT Progress");
    setFixedSize(400, 300);
    setWindowModality(Qt::ApplicationModal);

    QVBoxLayout *layout = new QVBoxLayout(this);

    // Display task title
    taskLabel = new QLabel("Running SALT Algorithm...", this);
    layout->addWidget(taskLabel);

    // Display image
    imageLabel = new QLabel(this);
    QPixmap pixmap("SALT.png"); // Ensure the image path is correct
    imageLabel->setPixmap(pixmap.scaled(100, 100, Qt::KeepAspectRatio));
    imageLabel->setAlignment(Qt::AlignCenter);
    layout->addWidget(imageLabel);

    // Display progress bar
    progressBar = new QProgressBar(this);
    progressBar->setRange(0, 100);
    layout->addWidget(progressBar);

    // Display additional information
    infoLabel = new QLabel("Please wait...", this);
    layout->addWidget(infoLabel);

    // Create cancel button
    cancelButton = new QPushButton("Cancel", this);
    layout->addWidget(cancelButton);
    connect(cancelButton, &QPushButton::clicked, this, &ProgressDialog::cancelProcess);

    process = new QProcess(this);

    connect(process, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(onProcessFinished(int, QProcess::ExitStatus)));

    connect(process, &QProcess::readyReadStandardOutput, this, &ProgressDialog::onProcessOutput);
    connect(process, &QProcess::readyReadStandardError, this, &ProgressDialog::onProcessErrorOutput);
    QString yamlPath = QDir::cleanPath(QCoreApplication::applicationDirPath() + "/../SALT/config.yaml");
    YAML::Node config = YAML::LoadFile(yamlPath.toStdString());

    QString filePath = QDir::cleanPath(QCoreApplication::applicationDirPath() + "/../SALT");
    QString env = QString::fromStdString(config["conda_sh_path"].as<std::string>());

    QString command = "source "+env+" && conda activate SALT && cd "+filePath +" && python -m main";

    process->start("bash", QStringList() << "-c" << command);  // macOS / Linux
}

ProgressDialog::~ProgressDialog() {
    if (process->state() == QProcess::Running) {
        process->kill();
    }
}

void ProgressDialog::onProcessOutput() {
    while (process->canReadLine()) { 
        QByteArray output = process->readLine();
        QString outputText = QString::fromUtf8(output).trimmed();


        if (outputText.startsWith("Info:")) {
            QMetaObject::invokeMethod(infoLabel, "setText", Qt::QueuedConnection, Q_ARG(QString, outputText.mid(5)));
        }

        if (outputText.startsWith("Progress:")) {
            int progressValue = outputText.mid(9).remove('%').toInt();
            QMetaObject::invokeMethod(progressBar, "setValue", Qt::QueuedConnection, Q_ARG(int, progressValue));
        }
    }
}

void ProgressDialog::onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus) {
    progressBar->setValue(100);
    taskLabel->setText("Finished!");
    if (exitStatus == QProcess::CrashExit) {
        taskLabel->setText("Task crushed!");
    }
    close();
}

void ProgressDialog::cancelProcess() {
    process->kill();
    taskLabel->setText("Task Canceled");
    close();
}
void ProgressDialog::onProcessErrorOutput() {
    QByteArray errorOutput = process->readAllStandardError();
    QString errorText = QString::fromUtf8(errorOutput).trimmed();
    qDebug() << "SALT info: " << errorText;
}
