import { useState } from "react"
import { useNavigate } from "react-router-dom";//Hook funtion access

export default function Search(){
    const [keyword,setKeyword]=useState(" ");

    const navigate=useNavigate();
    const searchHandeler = () =>{
            navigate('/search?keyword='+keyword)

    }

return <div className="input-group">
<input
  type="text"
  id="search_field"
  onChange={(e)=> setKeyword(e.target.value)}
  className="form-control"
  onBlur={searchHandeler}///automatic clear when search
  placeholder="Enter Product Name ..."
/>
<div className="input-group-append">
  <button onClick={searchHandeler} id="search_btn" className="btn">
    <i className="fa fa-search" aria-hidden="true"></i>
  </button>
</div>
</div>
}