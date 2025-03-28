#include "customdialog.h"

CustomDialog::CustomDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle("Notification");

    // 设置图片
    imageLabel = new QLabel(this);
    QPixmap pixmap("SALT.png");
    imageLabel->setPixmap(pixmap);
    imageLabel->setFixedSize(64, 64);
    imageLabel->setScaledContents(true);

    // 设置文本
    textLabel = new QLabel("You will run SALT.<br>Please confirm your config file.", this);

    // 按钮
    updateConfigBtn = new QPushButton("Update Config", this);
    runBtn = new QPushButton("Run", this);
    cancelBtn = new QPushButton("Cancel", this);

    // 绑定信号槽
    connect(updateConfigBtn, &QPushButton::clicked, this, &CustomDialog::openConfigFile);
    connect(runBtn, &QPushButton::clicked, this, &CustomDialog::accept);
    connect(cancelBtn, &QPushButton::clicked, this, &CustomDialog::reject);

    // 布局
    QHBoxLayout *topLayout = new QHBoxLayout;
    topLayout->addWidget(imageLabel);
    topLayout->addWidget(textLabel);

    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addWidget(updateConfigBtn);
    buttonLayout->addWidget(runBtn);
    buttonLayout->addWidget(cancelBtn);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addLayout(topLayout);
    mainLayout->addLayout(buttonLayout);
    setLayout(mainLayout);
}

void CustomDialog::openConfigFile() {
    QString filePath = QDir::cleanPath(QCoreApplication::applicationDirPath() + "/../SALT/config.yaml");
    QDesktopServices::openUrl(QUrl::fromLocalFile(filePath));
}
