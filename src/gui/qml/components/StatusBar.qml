import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Rectangle {
    id: root
    property bool processing: false
    
    width: parent.width
    height: 30
    color: "#e0e0e0"
    
    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 10
        anchors.rightMargin: 10
        spacing: 10
        
        Text {
            text: processing ? qsTr("Verarbeitung läuft...") : qsTr("Bereit")
            Layout.fillWidth: true
            elide: Text.ElideRight
        }
        
        BusyIndicator {
            visible: processing
            running: processing
            height: 24
            width: 24
        }
    }
} 