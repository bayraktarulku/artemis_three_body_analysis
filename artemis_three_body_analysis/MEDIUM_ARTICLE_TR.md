# Artemis II ve Üç Cisim Problemi: Bir Yazılımcının Yörünge Notları

50 yıl sonra, Nisan 2026'da, insanlar yeniden Ay'a bakacak. Artemis II, Apollo 17'den bu yana alçak Dünya yörüngesinin ötesine çıkan ilk insanlı görev olacak. Dört astronot, Ay'ın çevresinden dolanıp Dünya'ya geri dönecek. İniş yok. Bayrak dikmek yok. Ama bu "sadece bir tur" değil: doğru enerji, doğru faz ve doğru geometriyle fırlatılmış bir sistemin, yalnızca yerçekimi yardımıyla Ay'ın arkasından dönüp Dünya'ya geri düşmesi… Dinamik açıdan tam anlamıyla sofistike bir ballet.

Bu makale teknik bir rehber değil. Öğrenirken alınmış notlar. Yazılım geliştirirken duyulan dürtü de diyebiliriz: "Bunu kendim yapabilir miyim?" Eksik veya hatalı noktalar olabilir; düzeltmeler için şimdiden teşekkür ederim.

Odak noktam şu: Serbest dönüş (free-return) yörüngesinin arkasındaki matematik. Özellikle kısıtlı üç cisim problemi (CR3BP) ve bunu sayısal olarak çözmek için kullanılan yöntemler.

---

## Üç Cisim Problemi: Deterministik Ama Çözümsüz

Newton'un 1687'de formüle ettiği yerçekimi yasası iki cisim problemini analitik olarak çözebilir hale getiriyor. Dünya ve Güneş gibi iki kütleli bir sistemde hareket, kapalı form çözümlerle ifade edilebilir: elipsler (bağlı sistemler), parabol ve hiperbol (kaçan cisimler)… Temiz.

Ama üçüncü bir cisim sahneye çıkınca işler karışıyor. Sistem hala deterministiktir; yani Laplace'ın Şeytanı tüm başlangıç koşullarını bilseydi, önümüzdeki her kaotik adımı bir güzel planlayabilirdi. Yani klasik Newtoncu evren hala "Tanrı zar atmaz" modunda. Fakat Poincaré'nin 19. yüzyıl sonunda gösterdiği gibi, bu sistem kaotiktir: küçük farklar zamanla dramatik ayrışmalara yol açar.

İşte bu yüzden uzay görevleri tasarlarken bilim insanları, deterministik ama pratikte öngörülemez bu dinamiklerin üstünde dans etmek zorundadır. Ve dürüst olalım, olayın bütün heyecanı işte burada başlıyor.

---

## Kısıtlı Üç Cisim Problemi (CR3BP)

Artemis II'nin Dünya–Ay dinamiklerini ilk yaklaşımda anlamak için kullanılan model, Circular Restricted Three-Body Problem (CR3BP).

Basitleştirmeler şunlar:

- Dünya ve Ay birbirlerinin etrafında dairesel yörüngede döner.
- Uzay aracının kütlesi ihmal edilebilir (Dünya ve Ay'ı etkilemez).
- Koordinat sistemi Dünya–Ay hattıyla birlikte döner.
- Güneş ve diğer pertürbasyonlar ilk aşamada ihmal edilir.

Bu model fiziksel olarak eksiktir ama yapısal olarak güçlüdür. Çünkü sistemin topolojisini görmemizi sağlar: hangi enerji seviyesinde hangi bölgeler erişilebilir, hangi geçitler açık, hangileri kapalı?

---

## Dönen Referans Sistemi ve Geometri

Dünya–Ay hattıyla birlikte açısal hızla dönen bir koordinat sistemi seçiyoruz. Bu sistemde:

- Dünya sabit bir noktada,
- Ay sabit bir noktada,
- Uzay aracı hareketli.

Bu tercih önemli. Çünkü sabit referans sisteminde Dünya ve Ay sürekli yer değiştirirken, burada problem geometrik olarak sadeleşir.

Normalize edilmiş birimlerde Ay–Dünya mesafesi 1 kabul edilir. Kütle oranı:

```
μ = M_Ay / (M_Dünya + M_Ay) ≈ 0.01215
```

Bu küçük sayı sistemin asimetrisini temsil eder. Dünya baskındır ama Ay ihmal edilemez.

Uzay aracının konum vektörü **r** = (x, y, z) ve hız vektörü **v** = (vₓ, vᵧ, vᵤ) ile gösterilir. Newton'un ikinci yasasına göre hareket denklemleri, bu altı bilineni içeren birinci mertebeden altı diferansiyel denklem sistemine indirgenir.

Burada önemli kavram "efektif potansiyel"dir:

```
U(x,y,z) = (1-μ)/r₁ + μ/r₂ + ½(x² + y²)
```

Son terim merkezkaç etkisini temsil eder. Dönen referans sisteminin sonucu budur: Coriolis ve merkezkaç terimleri denklemlere eklenir.

Bu noktada sistem artık entegrasyona hazır bir formdadır. Analitik çözüm yok ama sayısal çözüm mümkündür.

![Artemis II Trajectory - Rotating Frame](artemis_trajectory_xy.png)
*Şekil 1: Dönen koordinat sisteminde Artemis II yörüngesi (xy düzlemi). Dünya ve Ay sabit konumda görünür. Yörünge Dünya'dan başlar, Ay'ın arkasından dolanır ve geri döner.*

---

## Jacobi Sabiti: Enerji Benzeri Bir Değişmez

CR3BP'nin en zarif özelliklerinden biri Jacobi sabitidir:

```
C = 2U - v²
```

Bu büyüklük yörünge boyunca sabit kalır. Klasik mekanikteki toplam enerjiye benzer ama dönen referans sistemine özgüdür.

Jacobi sabiti bize şunu söyler: Uzay aracı hangi "enerji seviyesinde" hareket ediyor? Bu enerji seviyesi, erişilebilir bölgeleri belirler. Zero-velocity surfaces denilen yüzeyler, aracın geçemeyeceği sınırları tanımlar.

Örneğin:

- Yüksek C → Dünya çevresine hapsolmuş yörünge
- C ≈ 3 civarı → Lagrange noktaları çevresinde geçitler
- Daha düşük C → Ay'a ulaşım mümkün

Serbest dönüş yörüngesi belirli bir Jacobi aralığında gerçekleşir. Yani mesele "Ay'a gitmek" değil; doğru enerji topolojisine yerleşmektir.

![Jacobi Constant Error Analysis](artemis_jacobi_error.png)
*Şekil 2: 10 günlük simülasyon boyunca Jacobi sabiti hata analizi. Ay geçişinde (~gün 5) hata artışı görülse de genel stabilite korunur.*

---

## Sayısal Entegrasyon: Gerçek İş Burada Başlıyor

Diferansiyel denklemler elimizde. Şimdi soru şu: Bunları 10 gün boyunca, hata birikimini kontrol ederek nasıl entegre edeceğiz?

Runge–Kutta ailesi burada devreye girer. Özellikle yüksek mertebeli, adaptif adım kontrollü yöntemler.

Artemis II gibi hassas görevlerde Dormand–Prince 8(5,3) (DOP853) gibi yöntemler tercih edilir. Bu yöntem:

- 8. mertebe doğruluk sağlar
- 5. mertebe gömülü çözümle hata tahmini yapar
- Adım boyutunu dinamik ayarlar

Temel fikir basit:

1. Mevcut durumdan türev hesapla.
2. Bir sonraki adımı tahmin et.
3. İki farklı mertebe çözümü karşılaştır.
4. Hata toleransın üzerindeyse adımı küçült.
5. Uygunsa kabul et ve devam et.

Yerel hata ~ O(h⁹), küresel hata ~ O(h⁸).

Ama kağıt üzerindeki mertebe bilgisi tek başına yeterli değil. Asıl test: Jacobi sabiti korunuyor mu?

---

## Hata Dinamikleri ve Ay Geçişi

Simülasyonlarda ilginç bir şey görürsünüz: Jacobi sabiti hatası Ay'a yakın geçişte artar. Bunun sebebi yerçekimi gradyanının hızla değişmesidir. Sistem stiff değildir ama hassastır.

Adaptif adım kontrolü burada kritik rol oynar. Ay'a yaklaşırken adım boyutu küçülür. Uzaklaşınca tekrar büyür.

10 günlük bir entegrasyonda Jacobi hatasının 10⁻⁹ seviyesinde kalması, modelin ve sayısal yöntemin sağlıklı çalıştığını gösterir.

Bu noktada yazılımcının aklına birkaç soru takılır:

- Enerji sapması (energy drift) oluyor mu?
- Singülariteye yaklaşma riski var mı?
- Step rejection oranı ne?

Kod, fizik kadar önemlidir.

---

## Serbest Dönüş Yörüngesi Nedir?

Serbest dönüş, motor kullanmadan Ay'ı dolanıp Dünya'ya geri dönen balistik bir yörüngedir. Apollo döneminde güvenlik mekanizması olarak tasarlandı.

Avantajları arasında, motor arızası durumunda bile geri dönüş imkanı, yakıt tasarrufu ve doğal dinamiklerden faydalanabilmek sayılabilir. Dezavantajlar ise geometriye dair kısıtlamalar, Ay inişi için uygun olmaması ve fırlatma zamanının kritik oluşudur.

Bu yörüngede Ay, uzay aracını bir tür yerçekimsel sapan gibi "bükerek" geri yönlendirir. Ama bu klasik gravity assist'ten farklıdır; enerji kazanımı değil, yön değişimi ön plandadır.

CR3BP çerçevesinden bakıldığında, araç belirli bir Jacobi seviyesinde L1/L2 geçitlerinden erişilebilir bir faz uzay bölgesine girer, Ay'ın arkasından dolanır ve Dünya potansiyel kuyusuna geri düşer.

Bu "geri düşme" ifadesi romantik değil; matematiksel olarak doğrudur.

---

## Simülasyon Çıktıları ve Fiziksel Kontroller

Tipik bir 10 günlük serbest dönüş simülasyonunda:

- Dünya perigee: ~400 km
- Ay periapsis: birkaç yüz kilometre
- Maksimum mesafe: ~380,000 km

![Artemis II 3D Trajectory](artemis_trajectory_3d.png)
*Şekil 3: Artemis II yörüngesinin üç boyutlu görünümü. Dönen referans sisteminde Dünya merkeze yakın, Ay sağda sabit konumda. Yörünge hafif eğimli (~1°) bir düzlemde ilerler.*

Kontrol listesi:

- Jacobi sabiti drift < 10⁻⁶ → kabul edilebilir
- Dünya ve Ay yarıçapı altına inme yok → çarpışma yok
- Entegrasyon başarıyla tamamlanmış → divergence yok

Yazılım tarafında bakılan metrikler:

- Fonksiyon çağrı sayısı
- Adım reddetme oranı
- Minimum step size

Bunlar fiziksel sonuç kadar önemlidir. Çünkü yanlış çalışan bir integratör, "doğru görünümlü ama yanlış" bir yörünge üretebilir.

---

## Modelin Sınırları

CR3BP güçlü bir yaklaşım sunar, ama eksiksiz değildir. İhmal edilen etkiler arasında Güneş çekimi, Dünya'nın J₂ terimi (ekvator şişkinliği), Ay'ın masconları ve Güneş radyasyon basıncı yer alır.

Gerçek görev tasarımında süreç genellikle şöyle işler:

1. CR3BP ile ilk taslak hazırlanır
2. Gerçek ephemeris verileri kullanılır
3. Pertürbasyonlar eklenir
4. Yüksek hassasiyetli entegratörlerle simülasyonlar çalıştırılır
5. Monte Carlo belirsizlik analizi yapılır

Artemis II gibi bir görev, binlerce simülasyon üzerinden optimize edilir. Yani CR3BP bir son değil, sadece başlangıçtır.

---

## Daha Geniş Perspektif: Lagrange Noktaları ve Gelecek

Bu model aynı zamanda Lagrange noktalarını anlamanın kapısını açar. L1 ve L2 çevresindeki halo yörüngeleri, Lunar Gateway gibi projelerin temelidir.

Artemis III ve sonrası görevlerde Ay yüzeyine iniş, NRHO (Near-Rectilinear Halo Orbit) ve düşük enerjili transferler devreye girecek. Bunların hepsi üç cisim dinamiklerinin uzantılarıdır.

Yani burada öğrendiğimiz diferansiyel denklemler, sadece bir Ay turunu değil; gelecekteki cislunar ekonomiyi de şekillendiriyor.

---

## Determinizm, Sayısal Hassasiyet ve Gerçeklik

Üç cisim problemi ilginç bir yerde durur. Sistem tamamen deterministiktir ama genel çözümü yoktur. Gelecek belirlenmiştir ama formül halinde yazılamaz.

Bu, mühendisliğin doğasına benzer.

CR3BP bize şunu öğretir:

- Doğru soyutlama, karmaşıklığı yönetilebilir kılar.
- Sayısal yöntemler teorinin uzantısıdır.
- Enerji korunumu, bir simülasyonun vicdanıdır.

Artemis II'nin serbest dönüş yörüngesi romantik bir "Ay turu" değil; enerji yüzeyleri üzerinde dikkatle seçilmiş bir başlangıç koşuludur.

En etkileyici kısım ise: Altı boyutlu bir durum vektörü, birkaç diferansiyel denklem ve iyi yazılmış bir integratör… Ve sonuçta Ay'ın arkasından dönüp eve gelen bir uzay aracı. Oldukça havalı.

Deterministik ama hassas. Basit varsayımlar ama karmaşık sonuçlar.

Belki de üç cisim problemiyle insan hayatı arasında beklenmedik bir paralellik var: Başlangıç koşullarını tam bilmek imkânsız, ama doğru modelle yönü anlayabiliyoruz.

---

## Kaynaklar

### Akademik Referanslar

1. **Szebehely, V.** (1967). *Theory of Orbits: The Restricted Problem of Three Bodies*. Academic Press.  
   → Üç cisim probleminin klasik ve temel başvuru kaynağı.

2. **Koon, W. S., Lo, M. W., Marsden, J. E., & Ross, S. D.** (2000). *Dynamical Systems, the Three-Body Problem and Space Mission Design*.  
   → CR3BP'nin uzay görevleri ve yörünge tasarımındaki uygulamaları.

3. **Hairer, E., Nørsett, S. P., & Wanner, G.** (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer.  
   → Runge–Kutta yöntemleri ve sayısal entegrasyon teorisi.

4. **Dormand, J. R., & Prince, P. J.** (1980). "A family of embedded Runge–Kutta formulae." *Journal of Computational and Applied Mathematics*, 6(1), 19–26.  
   → DOP853 yönteminin dayandığı gömülü Runge–Kutta formülleri.

5. **NASA** (2024). *Artemis II Mission Overview*.  
   → https://www.nasa.gov/artemis-ii

### Kod ve Simülasyon

Bu makalede bahsedilen tüm simülasyonlar açık kaynak Python kodu olarak GitHub'da mevcuttur:

**https://github.com/bayraktarulku/artemis_three_body_analysis**