import React from "react";

export interface Props {
    className: string;
    idSelf: string;
    children?: React.ReactNode;
  }


export default function DragSpace(props: Props) {
    function drag(ev) {
        ev.dataTransfer.setData("text", ev.target.id);
        ev.dataTransfer.dropEffect = "copy";
    }
    function allowDrop(ev) {
        ev.preventDefault();
        ev.dataTransfer.dropEffect = "copy";
      }
    
    function drop(ev) {
        ev.preventDefault();
        var data = ev.dataTransfer.getData("text");
        var copyimg = document.createElement("img");
        var original = document.getElementById(data);
        console.log(data)
        console.log(original)
        // copyimg.src = original.children[0].src;
        console.log(ev.target.children.length);
        if (ev.target.children.length > 0){
            ev.target.children.shift();
        }
        ev.target.appendChild(copyimg);
    }
    return (
        <div 
            className={props.className +  " " + "draggable"}
            id={props.idSelf}
            onDragStart={drag}
            draggable={true}
            onDrop={drop}
            onDragOver={allowDrop}
        >
        {props.children}
        </div>
    )
}