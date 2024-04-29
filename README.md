## Project Summary
Project Name : **Image Search Engine Toko Quick**
<br>
Environment : `image-search-tokoquick`
<br>
Algoritma Pendekatan : 
<br> 

* Pre-trained Keras [VGG19](https://keras.io/api/applications/vgg/#vgg19-function) for **feature extraction**, [reference](https://arxiv.org/abs/1409.1556).
* Pre-Trained Keras [ResNet50V2](https://keras.io/api/applications/resnet/#resnet50v2-function) for **feature extraction**, [reference](https://arxiv.org/abs/1409.1556).
* CBIR - Content-Based Image Retrieval, see [wikipedia](https://en.wikipedia.org/wiki/Content-based_image_retrieval).
* K-Nearest Neighbour [(KNN)]( https://github.com/ultralytics/yolov5), see [wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).

Folder Development : http://192.168.168.195/home/serverai/Project/features_extraction

<br>

## Quickstart
Clone repository ini, kemudian install seluruh dependensi yang terdapat dalam `requirements.txt`.
```bash
$ git clone http://gitlab.quick.com/artificial-intelligence/image-search-tokoquick.git     #clone
$ cd image-search-tokoquick
$ pip install -r requirements.txt       #install
```
<br>

## Dataset
Pengerjaan projek ini menggunakan kumpulan gambar produk, yang diambil dari website toko quick dengan teknik scraping. Tentunya dataset yang akan digunakan ini telah dibersihkan dari gambar-gambar noise.

<img title="gambar dalam toko quick" src="data/images/toko-quick.png">


<br>


## Train & Test 
### Extracting Features & Similarity Measurement
Proses utama dalam image retrieval berupa proses ekstraksi fitur gambar dan penghitungan jarak antar fitur. Kedua proses ini ditangani oleh `class features_extraction:` yang terdapat dalam file `main.py`.
```python

class features_extraction:
    def __init__(self):
        pass
    
    def get_product_id(self, data_path, product):
        ...
        return result

    def get_prediction(self, query_image, model, feature_list, filenames, preprocess_input):
        ...
        return df

```

<br>

## Model Optimation 
### Change Feature Extractor Model Configuration
File konfigurasi terletak pada `/utils/config.ini` . Anda dapat melakukan perubahan konfigurasi pada baris berikut.

```php
[model]
architecture = resnet50v2        # resnet50v2/vgg19
weights = imagenet
feature_list = features.pkl
filenames = filenames.pkl

[knn]
algorithm = brute
metric = cosine      # cosine/euclidean

[flask_api]
secret_key = tkquick_image_search
debug = False
session_type = filesystem
session_file_dir = temp
```

<br>

## Image Search Engine API
Projek ini menggunakan API yang berjalan di atas host http://ai.quick.com, dengan 2 endpoints utama sebagai berikut:
* Get API Info 
  ```bash 
  GET   /predict
  ```
* Image Search Engine
  ```bash 
  POST   /predict
  ```
<br>

## Source Code Explanation
## > Features Extraction Program 
**File Location : `main.py`** <br> Program ini menghandle proses *features extraction* dari setiap gambar, dan melakukan prediksi berdasarkan gambar yang diberikan. Terdapat dua kelas yaitu `loadModel()` yang berfungsi untuk memuat model dan *features*, dan kelas `features_extraction()` yang bertanggung jawab sebagai fungsi utama yaitu ekstraksi fitur dan prediksi ambar berdasarkan model yang dimuat sebelumnya.
### Define Config File
File pengaturan konfigurasi untuk program ini dapat ditemukan pada `utils/config.ini`.

```python
config = configparser.ConfigParser()
config.read(join(
                join(
                    dirname(abspath(__file__)), 'utils')
                , 'config.ini')
            )

....
```
### `class loadModel()`
- #### `begin()`
    Fungsi `begin()` melakukan warmup model yang ditentukan dalam `config['model']['architecture']`.
    ```python
        def begin(self, feature_list_path, filenames_path, architecture):
            
            # Loading Keras model berdasarkan opsi 'resnet50v2' atau 'vgg19' 
            if architecture == 'resnet50v2':
                from tensorflow.keras.applications.resnet_v2 import preprocess_input
                model = ResNet50V2(weights=config['model']['weights'],include_top=False,input_shape=(224,224,3))
            else:
                from tensorflow.keras.applications.vgg19 import preprocess_input
                model = VGG19(weights=config['model']['weights'],include_top=False,input_shape=(224,224,3))

            # Membuat model menjadi non-trainable sebagai features extractor
            model.trainable = False

            # Modifikasi model dengan menambah layer GlobalMaxPooling2D()
            model = Sequential([
                model,
                GlobalMaxPooling2D()
            ])

            # Menyimpan output dari after_model() ke dalam variabel feature_list dan filenames
            feature_list, filenames = self.after_model(feature_list_path, filenames_path)

            # Mengembalikan model, feature_list, filenames, preprocess_input sebagai output 
            return model, feature_list, filenames, preprocess_input

    ```
- #### `after_model()`
    Fungsi `after_model()` bertujuan untuk memuat fitur gamabr dan filepath dari hasil ekstraksi fitur yang disimpan sebagai pickle.

    ```python
        def after_model(self, feature_list_path, filenames_path):

            # Load pickle file ke dalam feature_list dan filenames
            feature_list = np.array(pickle.load(open(feature_list_path, 'rb')))
            filenames = pickle.load(open(filenames_path, 'rb'))
            
            # Membuat nama file baru dengan mengambil bagian terakhir path dengan .split()
            new_filenames = [data.split('/')[-1:] for data in filenames]
            
            return feature_list, new_filenames
    ```
<br>

### `class features_extraction()`
- #### `get_product_id()`
    Fungsi `get_product_id()` dalam kode yang diberikan bertanggung jawab untuk mencari ID produk berdasarkan nama produk dari `data/all_product.csv`. 

    ```python
        def get_product_id(self, data_path, product):
            df_data = pd.read_csv(data_path)
            nn = []
            for nama in df_data.values:
                # Menyamakan format penamaan produk
                new_nama = str(nama[1]).replace('/', '_').replace('  ', '').replace('"', '').replace('!', '')
                nn.append(new_nama)
            # Membuat kolom baru pada all_product.csv
            df_data['NewName'] = nn
            df_data.drop(columns=['PdNama'], inplace=True)
            # Mencari ID produk dari DataFrame berdasarkan nama produk yang diberikan dan mengembalikan hasilnya. 
            result = [data_num[0] for data_num in df_data.values if product == data_num[1]]
            return result
    ```
- #### `get_prediction()`
    Fungsi `get_prediction()` ini mengambil gambar senagai quey (`query_image`), menerapkannya pada model yang diberikan (model), dan kemudian membandingkan fiturnya dengan daftar fitur yang ada dalam f`eature_list` dengan bantuan algoritma KNN (NearestNeighbors). <br> Fungsi ini melakukan prediksi berdasarkan gambar yang diberikan, menemukan 10 tetangga terdekat dari daftar fitur yang ada, menghitung skor kesamaan, dan mengembalikan DataFrame yang berisi prediksi beserta ID produk, nama produk, dan skor kesamaannya.

    ```python
        def get_prediction(self, query_image, model, feature_list, filenames, preprocess_input):
            base_path = os.getcwd()
            data_path = os.path.join(base_path, 'data')
            product_path = os.path.join(data_path, 'all_product.csv')
            
            # Memproses gambar dalam query_image, menghasilkan output berupa normalized array dari hasil ekstraksi fitur pada gambar
            img = load_img(query_image,target_size=(224,224))
            img_array = img_to_array(img)
            expanded_img_array = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess_input(expanded_img_array)
            result = model.predict(preprocessed_img).flatten()
            normalized_result = result / norm(result)

            # memuat KNN dan mengaplikasikan .fit() dengan feature_list
            neighbors = NearestNeighbors(n_neighbors=10,algorithm=config['knn']['algorithm'],metric=config['knn']['metric'])
            neighbors.fit(feature_list)

            # Menghitung distance antara fitur query_image dan fitur dalam daftar feature_list
            distances,indices = neighbors.kneighbors([normalized_result])
    
            pred = []     # empty list
            
            # Mengambil 10 nearest neighbors, menghitung score kesamaan 
            for i in range(10):
                index = indices[0][i]
                distance = distances[0][i]
                result = filenames[index][0].split('.jpg')[0]
                score = round((1-(distance/1))*100, 2)
                product_name = result
                id_product = self.get_product_id(product_path, product=product_name)
                # Append result ke dalam pred[] list
                pred.append([id_product[0], product_name, score])
                
            # Membuat sebuah dataframe yang berisi `id_product`, `product_name`, `score`
            df = pd.DataFrame(pred, columns=['id_product', 'product_name', 'score'])
            return df
    ```
<br>

## > API Program 
**File Location : `app.py`** <br> Description .....
- #### Definne Model and Configuration 
    Potongan baris kode berikut memuat proses inisialisasi model beserta konfigurasi tiap model yang akan digunakan. Output yang dikembalikan adalah `model`, `feature_list`, `filenames`, dan `preprocess_input`, yang akan digunakan untuk proses selanjutnya seperti prediksi atau analisis.

    ```python
    base_path = dirname(abspath(__file__))
    model_path = join(base_path, 'models')
    # konfigurasi model resnet50v2
    if config['model']['architecture'] == 'resnet50v2':
        model_architecture_path = join(model_path, 'resnet50v2')
        feature_list_path = join(model_architecture_path, config['model']['feature_list'])
        filenames_path = join(model_architecture_path, config['model']['filenames'])
        trigger_model = loadModel()
        model, feature_list, filenames, preprocess_input = trigger_model.begin(feature_list_path, filenames_path, config['model']['architecture'])

    #konfigurasi model vgg19
    else:
        model_architecture_path = join(model_path, 'vgg19')
        feature_list_path = join(model_architecture_path, config['model']['feature_list'])
        filenames_path = join(model_architecture_path, config['model']['filenames'])
        trigger_model = loadModel()
        model, feature_list, filenames, preprocess_input = trigger_model.begin(feature_list_path, filenames_path, config['model']['architecture'])

    ```
- #### `predict()`
    Fungsi `predict()` merupakan bagian dari fungsi dalam API yang bertugas untuk menerima gambar, melakukan prediksi menggunakan model yang sudah dimuat sebelumnya, dan mengembalikan hasil prediksi dalam format JSON.

    ```python
    def predict():
        base_path = dirname(abspath(__file__))
        temp_path = join(base_path, 'temp')
        session['temp_file'] = join(temp_path, 'img_process.jpg')
        temp_json = join(temp_path, 'result.json')
        
        # Mengembalikan respon text jika request yang masuk != POST
        if request.method != "POST":
            return "API Image Search Tokoquick"
        
        # Memproses gambar yang didapat melalui request POST
        start_time = time.time()
        image_file = request.files["image"]
        image_bytes = image_file.read()
        with open(session['temp_file'], 'wb')  as outfile:
            outfile.write(image_bytes)

        # Menerapkan proses prediksi yang dimuat dalam kelas features_extraction()     
        main = features_extraction()
        preds = main.get_prediction(session['temp_file'], model, feature_list, filenames, preprocess_input)
        # Menyimpan hasil prediksi ke dalam JSON
        preds.to_json(temp_json, orient='records')
        f = open(temp_json)
        data_json = json.load(f)
        result = {
            "success": True,
            "preds": data_json,
            "time":round(time.time() - start_time, 2)
        }
        # Mengembalikan result
        result_json = jsonify(result)
        return result_json
    ```
   

<br>

### Testing Program
Selalu lakukan testing program langsung menggunakans serverai. Lakukan ssh ke server ai dengan serverai@192.168.168.195. Gunakan environment yang sesuai dengan penjelasan diatas. Running program python seperti biasa, python app.py. Pastikan saat itu port tidak terpakai oleh aplikasi lain. Jika program sudah berjalan, lakukan pengujian dengan mengirimkan gambar sample delam api.

Lihat dokumentasi api selengkapnya [disini](http://ai.quick.com/documentation/image-search-tokoquick/).