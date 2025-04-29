import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Rectangle {
    id: root
    color: "white"
    
    property var selectedImages: []
    property bool processing: false
    
    signal exportImages(var images)
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 20
        
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            
            Text {
                text: qsTr("Export mit Super-Resolution")
                font.pixelSize: 18
                font.bold: true
                Layout.fillWidth: true
            }
            
            ComboBox {
                id: formatSelector
                model: ["png", "tiff", "jpg"]
                Layout.preferredWidth: 100
            }
            
            ComboBox {
                id: modelSelector
                model: ["ESRGAN", "EDVR", "BASICVSR", "REAL_ESRGAN"]
                Layout.preferredWidth: 150
            }
            
            ComboBox {
                id: scaleSelector
                model: [2, 4]
                Layout.preferredWidth: 60
            }
            
            Button {
                text: qsTr("Exportieren")
                enabled: selectedImages.length > 0 && !processing
                
                onClicked: {
                    exportImages(selectedImages)
                }
            }
        }
        
        Item {
            Layout.fillWidth: true
            height: 40
            
            Text {
                anchors.centerIn: parent
                text: selectedImages.length > 0 ? 
                    qsTr("%1 Bilder für Export ausgewählt").arg(selectedImages.length) : 
                    qsTr("Keine Bilder ausgewählt. Wählen Sie zuerst Bilder in der Vorschau-Ansicht aus.")
                font.pixelSize: 14
                color: selectedImages.length > 0 ? "black" : "#888888"
            }
        }
        
        // Bilder anzeigen in einem Grid
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            
            GridView {
                id: grid
                anchors.fill: parent
                cellWidth: 200
                cellHeight: 200
                model: selectedImages
                delegate: imageDelegate
            }
        }
    }
    
    Component {
        id: imageDelegate
        
        Rectangle {
            width: grid.cellWidth - 10
            height: grid.cellHeight - 10
            border.color: "#dddddd"
            border.width: 1
            radius: 5
            
            Image {
                anchors.fill: parent
                anchors.margins: 5
                source: modelData
                fillMode: Image.PreserveAspectFit
                asynchronous: true
                
                // Platzhalter, falls Bild nicht geladen werden kann
                Rectangle {
                    anchors.fill: parent
                    color: "#f0f0f0"
                    visible: parent.status !== Image.Ready
                    
                    Text {
                        anchors.centerIn: parent
                        text: qsTr("Bild nicht verfügbar")
                        color: "#888888"
                    }
                }
            }
            
            // Dateiname anzeigen
            Rectangle {
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.margins: 5
                height: 30
                color: "#88000000"
                radius: 3
                
                Text {
                    anchors.fill: parent
                    anchors.margins: 5
                    text: {
                        var parts = modelData.split('/')
                        return parts[parts.length - 1]
                    }
                    color: "white"
                    elide: Text.ElideMiddle
                    font.pixelSize: 12
                    verticalAlignment: Text.AlignVCenter
                }
            }
        }
    }
} 