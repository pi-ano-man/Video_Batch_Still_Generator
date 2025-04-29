import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import QtQuick.Window

import "components"
import "views"

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1200
    height: 800
    title: qsTr("Video Batch Still Generator")
    
    // Fenstereigenschaften
    minimumWidth: 800
    minimumHeight: 600
    color: "#f5f5f5"  // Hintergrundfarbe
    
    // Statusleiste
    footer: StatusBar {
        id: statusBar
        processing: mainController.processing
    }
    
    // Hauptlayout mit Tabs für die verschiedenen Ansichten
    ColumnLayout {
        anchors.fill: parent
        spacing: 0
        
        // Toolbar mit Hauptaktionen
        Rectangle {
            Layout.fillWidth: true
            height: 60
            color: "#4a76a8"  // Dunkelblau für den Header
            
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 20
                anchors.rightMargin: 20
                spacing: 10
                
                Text {
                    text: qsTr("Video Batch Still Generator")
                    font.pixelSize: 20
                    color: "white"
                    font.bold: true
                }
                
                Item { Layout.fillWidth: true }  // Spacer
                
                Text {
                    text: qsTr("Person suchen:")
                    color: "white"
                    font.pixelSize: 14
                }
                
                ComboBox {
                    id: personSelector
                    model: mainController.get_people()
                    Layout.preferredWidth: 200
                    
                    onCurrentTextChanged: {
                        if (currentText) {
                            mainController.set_current_person(currentText)
                        }
                    }
                    
                    Component.onCompleted: {
                        if (count > 0) {
                            mainController.set_current_person(currentText)
                        }
                    }
                }
            }
        }
        
        // Tab-Leiste
        TabBar {
            id: tabBar
            Layout.fillWidth: true
            
            TabButton {
                text: qsTr("Videos")
            }
            TabButton {
                text: qsTr("Vorschau")
            }
            TabButton {
                text: qsTr("Export")
            }
        }
        
        // Tab-Inhalte
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabBar.currentIndex
            
            // Videos-Ansicht
            VideoListView {
                id: videoView
                onProcessVideo: function(path) {
                    mainController.process_video(path)
                }
                videos: mainController.videos
                processing: mainController.processing
            }
            
            // Vorschau-Ansicht
            ImagePreviewView {
                id: previewView
                selectedPerson: personSelector.currentText
                onSelectionChanged: function(images) {
                    mainController.set_selected_images(images)
                }
                processing: mainController.processing
            }
            
            // Export-Ansicht
            ExportView {
                id: exportView
                selectedImages: mainController.selected_images
                onExportImages: function(images) {
                    mainController.export_images(images)
                }
                processing: mainController.processing
            }
        }
    }
    
    // Dialog für Fehler
    MessageDialog {
        id: errorDialog
        title: qsTr("Fehler")
        
        Connections {
            target: mainController
            function onErrorOccurred(message) {
                errorDialog.text = message
                errorDialog.open()
            }
        }
    }
} 