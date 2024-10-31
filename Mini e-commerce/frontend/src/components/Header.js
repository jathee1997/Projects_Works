import {Link} from "react-router-dom";
import Search from "./Search";


export default function Header({cartItems}){
    return<nav className="navbar row">
      <div className="col-12 col-md-3">
      <div className="navbar-brand">
        <Link to ="/" ><img width="100px" height="50px" src="/images/logo.png" alt="Logo" /></Link>{/* this is when click logo show index page*/}
        {/*<img width="100px" height="50px" src="/images/logo.png" alt="Logo" />*/}
      </div>
      </div>

    <div className="col-12 col-md-6 mt-2 mt-md-0">
      <div className="input-group">
        <Search/>
      </div>
    </div>

    <div className="col-12 col-md-3 mt-4 mt-md-0 text-center">

      <Link to ={"/cart"}>
      <span id="cart" className="ml-3">Cart</span>
      <span className="ml-1" id="cart_count">{cartItems.length}</span>
      </Link>
      
    </div>
  </nav>
}