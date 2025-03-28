#ifndef PROGRESSDIALOG_H
#define PROGRESSDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QProcess>
#include <QCoreApplication>
#include <QFileDialog>
#include <yaml-cpp/yaml.h>

class ProgressDialog : public QDialog {
    Q_OBJECT

public:
    explicit ProgressDialog(QWidget *parent = nullptr);
    ~ProgressDialog();

private slots:
    void onProcessOutput();
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void cancelProcess();
    void onProcessErrorOutput();

private:
    QLabel *taskLabel;
    QLabel *imageLabel;
    QProgressBar *progressBar;
    QLabel *infoLabel;
    QPushButton *cancelButton;
    QProcess *process;
};

#endif // PROGRESSDIALOG_H
