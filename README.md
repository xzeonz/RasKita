Dibaca ya, ga baca gue sumpahin

Machine Learning: Klasifikasi Jenis Anjing dan Kucing

1. Pendahuluan:
    Proyek ini didesain untuk melakukan klasifikasi jenis anjing dan kucing peliharaan menggunakan machine learning -> deep learning dengan PyTorch untuk backend dan React.js untuk frontend. Dalam proyek ini, sistem memberikan akses untuk upload gambar anjing dan kucing lalu sistem memberikan hasil prediksi klasifikasi beserta tingkat presentase tingkat kemiripan.

2. Pendekatan pada Machine Learning:
a. Dataset
    Dataset yang digunakan dalam proyek ini adalah dataset yang telah disediakan oleh kaggle Oxford-III Pet Dataset, menggunakan 37 jenis gambar anjing dan kucing dengan 1 jenisnya berisi kurang lebih 200 gambar dengan format jpg.

b. Pendekatan model
    Model yang kami gunakan adalah ResNet-50, CNN (Convolutional Neural Network) yang telah dilatih di ImageNet. Lapisan terakhir dilakukan penyesuaian untuk mengklasifikasikan jenis anjing dan kucing.

c. Proses Pelatihan

    1. Preproses data
        - Gambar diresize menjadi 224x224.
        - Normalisasi menggunakan mean = [0.485, 0.456, 0.406] dan std=[0.229, 0.224, 0.225]. ==> dapat nilainya darimana? ini dari dataset ImageNet yang sudah dihitung sebelumnya berdasarkan distribusi wara yang ada di dataset.

            => Gimana nilainya dihitung?
                - Ini sebenarnya udah darisana dihitung sama peneliti terkait menghitung rata-rata mean dan standar deviasi std dari semua gambar yang ada di dataset ImageNet. Jadi ini udah nilai standar di banyak model deep learning yang ditrain pakai ImageNet, termasuk ResNet, VGG, MobileNet, dan EfficientNet.

            => Alasan digunakan:
                - Ketika menggunakan model yang sudah dipretrained pada ImageNet (layaknya ResNet-50), sebaiknya harus menyesuaikan input agar sesuai dengan distribusi data ImageNet.
            
            !! CAUTION !!
                Kalau tidak menormalkan gambar dengan nilai, model bisa malfungsi prediksi karena distribusi datanya tidak cocok dengan data ketika model dilatih.

            => Gimana kalo dataset bukan dari ImageNet?
            Ya buat sendirilah kocak ರ╭╮ರ
            Nah kalo buat sendiiri, lebih baik ngitung mean dan std-nya dari dataset dari kita sendiri
                
                Contoh kode sheet buat menghitung mean dan std dari dataset sendiri:
                ___________________________________________________________________________________________
                import numpy as np                                      
                from torchvision import datasets, transforms

                # Load dataset tanpa normalisasi
                dataset = datasets.ImageFolder("path/to/dataset", transform=transforms.ToTensor())

                # Hitung mean dan std per channel
                images = torch.stack([img[0] for img in dataset])  # Ambil semua gambar dalam bentuk tensor
                mean = images.mean(dim=[0, 2, 3])  # Rata-rata tiap channel
                std = images.std(dim=[0, 2, 3])  # Standar deviasi tiap channel

                print(f"Mean: {mean.tolist()}, Std: {std.tolist()}")
                ___________________________________________________________________________________________

        - Augmentasi digunakan untuk upgrade performa model:
            - Random cropping, flipping, dan rotasi untuk membuat model lebih robust.
            - Color jitter untuk meningkatkan variasi warna pada dataset.
            - Random erasing untuk support model belajar fitur yang lebih umum

    2. Pelatihan Model
        - Modifikasi fc (Fully Connected) pada ResNet-50 guna menyesuaikan jumlah kelas (37 class anjing dan kucing).
        - Model menggunakan scheduler ReduceLROnPlateau guna penyesuaian learning rate berdasarkan performa validasi.
        - Early stopping diterapkan dengan patience 5 untuk mencegah overfitting.
        - Gradient clipping diterapkan guna menghindari exploding gradients.

    3. Penjelasan Kode:
    =====================================================================================================================
    .\Backend\data\train_model.py:

        1. Import Library:
            _______________________________________________
            import torch
            import torchvision.transforms as transforms
            from torchvision.datasets import ImageFolder
            from torch.utils.data import DataLoader, random_split
            import torch.nn as nn
            import torch.optim as optim
            from torchvision import models
            import os
            from sklearn.metrics import accuracy_score
            import numpy as np
            import platform
            _______________________________________________

            Penjelasan:
                - PyTorch                           -> digunakan karena ini merupakan framework terkuat untuk deep learning.
                - tochvision.transforms             -> digunakan untuk augmentasi dan preprocessing gambar.
                - ImageFolder                       -> memungkinkan pembacaan dataset yang terstruktur dalam folder berdasarkan class.
                - random_split                      -> digunakan untuk memecah dataset menjadi train dan validation agar model bisa dieval dengan akurat.
                - nn dan optim                      -> modul PyTorch untuk membuat model dan optimasi.
                - models                            -> digunakan guna membuat model ResNet-50 yang sudah pretrained.
                - os                                -> digunakan untuk operasi sistem file.
                - accuracy_score dari sklearn.metrics   -> untuk menghitung akurasi.
                - numpy                             -> untuk manipulasi data numerik.
                - platform                          -> untuk deteksi sistem operasi like windows, linux, macos, dll.

        2. Hyperparameter & Device Selection
            _______________________________________________
            data_dir = "./backend/data/images"
            batch_size = 64  # Diperbesar untuk memanfaatkan GPU
            learning_rate = 0.0001
            epochs = 30  # Ditambah dengan early stopping
            patience = 5  # Early stopping parameter
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _______________________________________________

            Penjelasan:
                - batch_size = 64                       -> Ditingkatkan guna memanfaatan GPU dengan efisien. Batch besar mempercepat konvergensi, tapi perlu lebih banyak storage GPU.
                - learning_rate = 0.0001                -> Learning rate kecil membantu model belajar dengan stabil dan menghindari overshoot.
                - epochs = 30                           -> Batas maks iterasi pelatihan. Early stopping akan menghentikan lebih awal jika tidak ada peningkatan performa.
                - patience = 5                          -> Jika akurasi validasi tidak meningkat selama 5 epoch berturut2, maka pelatihan dihentikan.
                - device = "cuda"                       -> Model akan menggunakan GPU jika tersedia untuk mempercepat proses pelatihan.

        3. Transformasi dan Augmentasi Data
            _______________________________________________
            train_transform = transforms.Compose([
                transforms.Resize((256, 256)),  # Lebih besar untuk cropping
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),  # Regularisasi tambahan
            ])

            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            _______________________________________________

            Penjelasan:
                - Augmentasi dilakukan hanya pada pelatihan data untuk meningkatkan generalisasi model.
                - Resize(256, 256)                          -> Memperbesar gambar untuk kemudian dicrop ke ukuran 224 x 224.
                - RandomHorizondalFlip()                    -> Membantu model mengenali objek meskipun dicerminkan secara horizontal.
                - RandomResizedCrop(224, scale=(0.8, 1.0))  -> Memotong bagian gambar secara acak untuk memperkenalkan variasi.
                - ColorJitter()                             -> Mengubah kecerahan, saturasi, dan kontras untuk membuat model lebih tahan terhadap perubahan pencahayaan.
                - RandomErasing()                           -> Menghapus sebagian kecil gambar secara acak agar model tidak terlalu bergantung pada fitur tertentu.

        4. Membuat dataset dan membagi data
            ___________________________________________
            dataset = ImageFolder(data_dir)

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_dataset.dataset.transform = train_transform
            val_dataset.dataset.transform = val_transform
            ___________________________________________

            Penjelasan:
                - ImageFolder(data_dir)                     -> Dataset dimuat berdasarkan struktur folder yang berisi sub-folder untuk setiap kelas.
                - random_split()                            -> Memisahkan dataset menjadi 80% untuk training dan 20% untuk validation.
                - Transformasi dilakukan setelah pemisahan dataset -> untuk mencegah data leak (ketika informasi dari validation data bocor ke training).

        5. DataLoader untuk training dan validation
            ____________________________________________
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, 
                num_workers=0 if is_windows else 4, pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, 
                num_workers=0 if is_windows else 4, pin_memory=True
            )
            _____________________________________________

            Penjelasan:
                - shuffle = True   , pada training      -> Agar data acak di setiap epoch membantu model belajar dengan baik.
                - num_workers = 4                       -> Mempercepat loading data, tetapi diset 0 di windows guna mencegah error multiprocessing.
                - pin_memory = True                     -> Meningkatkan efisiensi saat transfer data ke GPU.

        6. Model ResNet-50 dan Fine-Tuning
            ______________________________________________
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

            for param in model.parameters():
                param.requires_grad = False

            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, len(dataset.classes))
            )
            model = model.to(device)
            ______________________________________________

            Penjelasan:
                - ResNet-50 digunakan karena performa yang baik di bidang klasifikasi gambar.
                - requires_grad = False                 -> Semua layer dibekukan agar model tidak kehilangan fitur yang telah dipelajari dari ImageNet.
                - FC (Fully Connected) layer dimodif untuk menyesuaikan dengan jumlah kelas dataset (37 class anjing dan kucing).
                - Batch Normalization dan dropout ditambahkan untuk meningkatkan stabilitas dan mengurangi overfitting.

        7. Optimizer, Scheduler, dan Loss Function
            _____________________________________________
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            _____________________________________________
            
            Penjelasan:
                - AdamW digunakan karena lebih baik dalam regularisasi dibandingkan Adam.
                - Weight decay (0.001)                   -> Mencegah overvitting dengan menekan bobot yang terlalu besar.
                - ReduceLROnPleteau                     -> Menurunkan learning rate jika akurasi tidak meningkat selama 2 epoch.
                - Label Smoothing (0.1)                 -> Mengurangi overconfidence model, membuat model lebih general.

        8. Training Loop dan Early Stopping
            _____________________________________________
            best_acc = 0.0
            no_improve = 0
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0

                for images, labels in train_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
            ______________________________________________

            Penjelasan:
                - Loop utama menjalankan training selama epochs.
                - optimizer.zero_grad()                 -> reset gradient sebelum backward pass.
                - loss.backward()                       -> Backpropagation untuk update parameter.
                - optimizer.step()                      -> Update bobot model.
    ==============================================================================================================================

    .\Backend\data\app.py :

        1. Import Library
            ___________________________________________
            from flask import Flask, request, jsonify
            from flask_cors import CORS
            from PIL import Image
            import torch
            import torch.nn as nn
            import torchvision.transforms as transforms
            from torchvision import models, datasets
            import io
            import os
            ____________________________________________

            Penjelasan:
                - Flask dan Flask-CORS
                    a. Flask: Framework untuk membangun API.
                    b. CORS: Mengizinkan akses dari frontend berbasis react atau apps lain ke API.
                - PIL (Pillow)                          -> digunakan untuk proses gambar sebelum dikirim ke model.
                - torch, torchvision
                    a. torch    -> Untuk bekerja dengan model deep learning di PyTorch.
                    b. torchvision.transforms   -> Untuk melakukan transform (resize, normalisasi) pada gambar yang masuk.
                    c. model    -> membuat model ResNet-50.
                    d. datasets -> untuk membaca dataset dan mendapatkan nama kelas.
                - io dan os
                    a. io.BytesIO() -> membantu membaca gambar dari request.
                    b. os   -> mengatur path file model yang telah dilatih.

        2. Inisialisasi Flask
            ____________________________
            app = Flask(__name__)
            CORS(app)
            _____________________________

            Penjelasan:
                - app = Flask(__name__)         -> Membuat instance Flask sebagai API.
                - CONS(app)                     -> Mengaktifkan Cross-Origin Resource Sharing agar API bisa diakses dari berbagai sumber termasuk frontend react.

        3. Menentukan Device (CPU/GPU)
            ___________________________
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ___________________________

            Penjelasan:
                - Jika GPU tersedia, model menggunakan CUDA untuk akselerasi.
                - Jika tidak, model akan berjalan di CPU.

        4. Membuat Dataset dan Nama Kelas
            ___________________________
            data_dir = "./backend/data/images"
            dataset = datasets.ImageFolder(data_dir)
            class_names = dataset.classes
            ___________________________

            Penjelasan:
                - Dataset dari data_dir diload dengan ImageFolder() yang secara otomatis membaca nama sub-folder sebagai label kelas.
                - class_names = dataset.classes     -> menyimpan daftar nama kelas yang nanti digunakan untuk hasil prediksi.

        5. Definisi Transformasi Gambar
            ____________________________
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            _____________________________

            Penjelasan:
                - Transformasi ini harus sesuai dgn yg digunakan saat training.
                - Resize (224, 224)     -> Gambar diubahh ke ukuran 224x224 agar sesuai dengan ResNet-50.
                - ToTensor()            -> Konversi gambar jadi tensor biar bisa dipake sama PyTorch.
                - Normalize()           -> Standarisasi nilai pixel agar sesuai dengan distribusi ImageNet (dataset pretraining ResNet-50)

        6. Definisi Struktur Model
            _______________________________
            def create_model(num_classes):
            model = models.resnet50(weights=None)
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            return model
            _______________________________

            Penjelasan:
                - Model yang digunakan adalah ResNet-50, tapi FC (Fully Connected) layer dimodif biar disesuaikan dgn jumlah class.
                - Arsitektur tambahan pada FC layer:
                    a. Linear(2048 -> 1024) -> BatchNorm -> ReLU -> Dropout(0.5)
                    b. Linear(1024 -> 512) -> BatchNorm -> ReLU -> Dropout(0.3)
                    c. Linear(512 -> num_classes) -> Layer terakhir menentukan jumlah kelas yang diprediksi.
                - Dropout       -> Mencegah overfitting dengan mengabaikan beberapa neuron secara random selama training.
                - BatchNorm     -> Meningkatkan stabilitas training.

        7. Membuat Model yang Telah Dilatih
            _______________________________________
            model = create_model(len(class_names))
            model_path = os.path.join("./backend", "cat_breed_classifier.pth")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            ______________________________________

            Penjelasan:
                - create_model(len(class_names))    -> Membuat model dengan jumlah kelas sesuai dataset.
                - torch.load(model_path, map_location=device)
                    a. Membuat bobot model yang telah dilatih (fileName.pth).
                    b. map_location=device -> Ketika model dilatih di GPU tapi pengen dipake di CPU, ini memastikan kompatibilitas.
                - model.to(device)          -> Model dikirim ke GPU (jika tersedia) atau CPU untuk interfensi.
                - model.eval()              -> Model disetel ke mode eval (tdak akan update bobot lagi).
        
        8. Endpoint Root(/)
            _______________________________________
            @app.route('/')
            def home():
                return "Cat Breed Classification API - Use /predict endpoint for classification"
            _______________________________________

            Penjelasan:
            - Saat API diakses melalui browser / postman, hal ini akan menampilkan pesan bahwa API sudah berjalan.

        9. Endpoint /predict untuk Prediksi Gambar
            _______________________________________
            @app.route('/predict', methods=['POST'])
            def predict():
                if 'file' not in request.files:
                    return jsonify({'error': 'No file uploaded'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'Empty file name'}), 400
            ________________________________________

            Penjelasan:
            - Mengecek apakah file ada di dalam request?
                a. Jika tidak ada file  -> API return error 400
                b. Jika file kosong -> API return error 400

        10. Membaca dan Memproses Gambar
            _______________________________________
            try:
                # Read and preprocess image
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image = val_transform(image).unsqueeze(0).to(device)
            ________________________________________

            Penjelasan:
                - file.read()   -> Membca file gambar dlm bentuk bit.
                - Image.open(io.BytesIO(image_bytes)).convert('RGB')    -> Membuka gambar dan memastikan dlm format RGB.
                - Transform gambar (val_transform) dilakukan agar format cocok dgn model.
                unsqueeze(0)    -> nambah dimensi batch (karena model menharapkan batch input)
                - to(device)    -> mengirim gambar ke GPU jika tersedia.

        11. Melakukan Prediksi
            _______________________________________
            # Prediction
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top prediction
            top_prob, top_catid = torch.max(probabilities, dim=0)
            prediction = {
                'breed': class_names[top_catid.item()],
                'confidence': f"{top_prob.item()*100:.2f}%"
            }
            
            return jsonify({'prediction': prediction})
            _______________________________________

            Penjelasan:
                - torch.no_grad()   -> Menghindari perhitungan gradient (lebih efisien utk interfensi)
                - soft,ax(outputs[0])   -> Mengubah output model menjadi probabilitas
                - torch.max(probabilities, dim=0)   -> Mendapatkan indeks kelas dengan probabilitas tertinggi
                - Format predict result dlm JSON
                    a. Nama ras kucing
                    b. confidence score (dalam persen)

        12. Jalanin API
            _______________________________________
            if __name__ == '__main__':
            app.run(host='0.0.0.0', port=5000, debug=True)
            _______________________________________

            Penjelasan:
            - Jalanin API di port 5k
            - host='0.0.0.0'    -> API bisa diakses dr jaringan lain
            - debug=True        -> Nampilin error jika ada kesalahan
    =============================================================================================================

    cara setup react js
    1. npm create vite
    2. pilih react
    3. namain proyeknya apa
    4. ikutin 3 prosedur yg udh disediain sama vitenya di cmd
    5. dan edat edit

    .\frontend\src\app.jsx
    >>desain sendiri lah ya<<





