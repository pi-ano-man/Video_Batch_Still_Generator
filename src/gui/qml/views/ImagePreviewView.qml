import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts

Rectangle {
    id: root
    color: "white"
    
    property string selectedPerson: ""
    property bool processing: false
    property var selectedImages: []
    
    signal selectionChanged(var images)
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 20
        
        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            
            Text {
                text: qsTr("Bildvorschau: ") + (selectedPerson ? selectedPerson : qsTr("Keine Person ausgewählt"))
                font.pixelSize: 18
                font.bold: true
                Layout.fillWidth: true
            }
            
            Button {
                text: qsTr("Alle wählen")
                enabled: grid.count > 0 && !processing
                onClicked: {
                    // Wähle alle Bilder aus
                    var allImages = []
                    for (var i = 0; i < imageModel.count; i++) {
                        allImages.push(imageModel.get(i).source)
                    }
                    selectedImages = allImages
                    selectionChanged(allImages)
                }
            }
            
            Button {
                text: qsTr("Auswahl aufheben")
                enabled: selectedImages.length > 0 && !processing
                onClicked: {
                    selectedImages = []
                    selectionChanged([])
                }
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
                model: imageModel
                delegate: imageDelegate
                
                Text {
                    anchors.centerIn: parent
                    visible: grid.count === 0
                    text: qsTr("Keine Bilder gefunden.\nWählen Sie eine Person aus und starten Sie die Verarbeitung.")
                    horizontalAlignment: Text.AlignHCenter
                    color: "#888888"
                    font.pixelSize: 14
                }
            }
        }
    }
    
    // Simuliertes Modell für die Demo
    ListModel {
        id: imageModel
        
        // Wird in der echten Implementierung durch Daten aus dem Backend gefüllt
        // ListElement { source: "image1.jpg"; selected: false }
        // ListElement { source: "image2.jpg"; selected: false }
        // ListElement { source: "image3.jpg"; selected: false }
    }
    
    Component.onCompleted: {
        // Dieses Modell wird in der vollständigen Implementierung vom Controller gefüllt
        // Für die Demo verwenden wir Platzhalterdaten
        imageModel.clear()
        // In der echten Implementierung: Lade Bilder für die ausgewählte Person
    }
    
    onSelectedPersonChanged: {
        // In der echten Implementierung: Lade Bilder für die neue Person
        console.log("Person geändert: " + selectedPerson)
    }
    
    Component {
        id: imageDelegate
        
        Rectangle {
            width: grid.cellWidth - 10
            height: grid.cellHeight - 10
            border.color: isSelected() ? "#4a76a8" : "#dddddd"
            border.width: isSelected() ? 3 : 1
            radius: 5
            
            function isSelected() {
                return root.selectedImages.indexOf(model.source) !== -1
            }
            
            Image {
                anchors.fill: parent
                anchors.margins: 5
                source: model.source
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
            
            MouseArea {
                anchors.fill: parent
                enabled: !root.processing
                
                onClicked: {
                    var selection = [...root.selectedImages]
                    var index = selection.indexOf(model.source)
                    
                    if (index === -1) {
                        // Hinzufügen zur Auswahl
                        selection.push(model.source)
                    } else {
                        // Entfernen aus der Auswahl
                        selection.splice(index, 1)
                    }
                    
                    root.selectedImages = selection
                    root.selectionChanged(selection)
                }
            }
            
            // Selektions-Indikator
            Rectangle {
                anchors.top: parent.top
                anchors.right: parent.right
                anchors.margins: 5
                width: 24
                height: 24
                radius: 12
                color: isSelected() ? "#4a76a8" : "transparent"
                border.color: "#dddddd"
                border.width: 1
                visible: isSelected()
                
                Text {
                    anchors.centerIn: parent
                    text: "✓"
                    color: "white"
                    font.pixelSize: 14
                    font.bold: true
                }
            }
        }
    }
} 