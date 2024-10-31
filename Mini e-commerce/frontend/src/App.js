import './App.css';
import Home from './pages/Home';
import Header from './components/Header'  
import Footer from './components/Footer'  
import { BrowserRouter as Router,Routes,Route  } from 'react-router-dom';//this package use connect to product card // link to 
import ProductDetails from './pages/ProductDetails';
import { useState } from 'react';

import {ToastContainer} from 'react-toastify'; 
import 'react-toastify/dist/ReactToastify.css';
import Cart from './pages/Cart';
function App() {

  const [cartItems,setCartItems]=useState([])
  return (
    <div className="App">
        
            <Router>
              <div>
                <ToastContainer theam='dark' position='top-center'/>
              <Header cartItems={cartItems}/>
              <Routes>
            <Route path="/" element={<Home />}/>
            <Route path="/search" element={<Home />}/>
            <Route path="/product/:id" element={<ProductDetails cartItems={cartItems} setCartItems={setCartItems}/>}/>{/*id  value throw to ProductDetails*/}
            <Route path="/cart" element={<Cart cartItems={cartItems} setCartItems={setCartItems}/>}/>
        </Routes>
        {/*id  value throw to Cart*/}
              </div>
              
      </Router>
      {/* <Home/> befor link path*/}
      <Footer/>
    </div>
  );
}

export default App;
