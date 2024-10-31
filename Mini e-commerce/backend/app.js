const express=require('express');
const app =express();
const dotenv= require('dotenv');
const path = require('path');
dotenv.config({path:path.join(__dirname,'config', 'config.env')})
const connectDatabase=require('./config/connectDatabase');

const cors=require('cors');// must install CORS policy
const  products=require('./routes/product');
const  orders=require('./routes/order');

connectDatabase();

app.use(express.json());//this is midle ware//post data oreder router
app.use(cors())//this is midle ware//response data error control// response header handler
app.use('/api/v1/',products);
app.use('/api/v1/',orders);
app.listen(process.env.PORT,()=>{
    console.log(`Server listening to Port ${process.env.PORT} in ${process.env.NODE_ENV}`)
})