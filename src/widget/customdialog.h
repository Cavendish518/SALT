#ifndef CUSTOMDIALOG_H
#define CUSTOMDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QDesktopServices>
#include <QUrl>
#include <QCoreApplication>

class CustomDialog : public QDialog {
    Q_OBJECT

public:
    explicit CustomDialog(QWidget *parent = nullptr);

private slots:
    void openConfigFile();

private:
    QLabel *imageLabel;
    QLabel *textLabel;
    QPushButton *updateConfigBtn;
    QPushButton *runBtn;
    QPushButton *cancelBtn;
};

#endif // CUSTOMDIALOG_H
