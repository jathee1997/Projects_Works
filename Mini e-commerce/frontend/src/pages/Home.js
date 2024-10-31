import { Fragment,useEffect,useState } from 'react';
import ProductCard from '../components/ProductCard';
import { useSearchParams } from 'react-router-dom';

export default function Home() {

  const [products, setProducts]=useState([]);
  const [searchParams, setSearchParams]=useSearchParams();//this is support to search

  useEffect(()=>{         //UseEffect hook
        fetch(process.env.REACT_APP_API_URL+'/products?'+searchParams) // Access url from backend to frontend end has .env.file ///it is request
        .then(res=>res.json())//response get ,change formate json
        .then(res => setProducts(res.products))//catch this then
  },[searchParams])//call search params
  return <Fragment>
    

      <h1 id="products_heading">Latest Products</h1>

      <section id="products" className="container mt-5">
        <div className="row">
          {products.map(product=><ProductCard product={product}/>)} {/*product={product} this is probes ,paasing to ProductCard*/}
          {/*<ProductCard/>  ------repeate componet is created seperate*/}
        </div>
      </section>
      
    </Fragment>
  
}
