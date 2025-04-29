import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Dialog {
    id: root
    
    property alias text: messageText.text
    
    modal: true
    standardButtons: Dialog.Ok
    x: (parent.width - width) / 2
    y: (parent.height - height) / 2
    width: Math.min(parent.width - 50, 400)
    
    contentItem: ColumnLayout {
        spacing: 20
        
        Text {
            id: messageText
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            horizontalAlignment: Text.AlignHCenter
        }
    }
} 