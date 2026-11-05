"""
=============================================================================
Module: classification.py
Pipeline Steps 7 & 8 — Klasifikasi Rule-Based & Rekomendasi Orthotics
=============================================================================
Fungsi utama:
  - classify_arch()       : Klasifikasi berdasarkan threshold Cavanagh & Rodgers.
  - check_asymmetry()     : Deteksi asimetri antara kaki kiri dan kanan.
  - get_recommendation()  : Rekomendasi orthotics berdasarkan klasifikasi.
  - generate_report()     : Generate laporan lengkap per subjek.
=============================================================================
"""

# Threshold Literatur Cavanagh & Rodgers (1987) untuk Jejak Tinta Kertas (Ink Print)
CAVANAGH_THRESHOLDS = {
    'high_arch': 0.21,   # AI <= 0.21
    'flat_foot': 0.26,   # AI >= 0.26
}

# Threshold Terkalibrasi untuk Sensor Tekanan Plantar Digital (Electronic Mat)
SENSOR_CALIBRATED_THRESHOLDS = {
    'high_arch': 0.22,   # AI <= 0.22
    'flat_foot': 0.33,   # AI >= 0.33
}


def classify_arch(arch_index: float, 
                  high_thresh: float = None, 
                  flat_thresh: float = None,
                  mode: str = 'calibrated') -> str:
    """
    Klasifikasi tipe lengkungan kaki berdasarkan Arch Index.
    
    Mode:
      - 'calibrated': Ambang batas terkalibrasi khusus sensor tekanan plantar digital
                      (High Arch <= 0.22, Normal 0.22-0.33, Flat Foot >= 0.33).
      - 'cavanagh'  : Ambang batas literatur klasik Cavanagh & Rodgers 1987
                      (High Arch <= 0.21, Normal 0.21-0.26, Flat Foot >= 0.26).
    """
    if arch_index < 0:
        return 'Invalid'
        
    if high_thresh is None or flat_thresh is None:
        if mode == 'cavanagh':
            high_thresh = CAVANAGH_THRESHOLDS['high_arch']
            flat_thresh = CAVANAGH_THRESHOLDS['flat_foot']
        else:
            high_thresh = SENSOR_CALIBRATED_THRESHOLDS['high_arch']
            flat_thresh = SENSOR_CALIBRATED_THRESHOLDS['flat_foot']
            
    if arch_index <= high_thresh:
        return 'High Arch'
    elif arch_index >= flat_thresh:
        return 'Flat Foot'
    else:
        return 'Normal'


def check_asymmetry(ai_left: float, ai_right: float,
                    threshold: float = 0.05) -> dict:
    """
    Deteksi asimetri bilateral antara kaki kiri dan kanan.
    
    Args:
        ai_left: Arch Index kaki kiri.
        ai_right: Arch Index kaki kanan.
        threshold: Batas perbedaan AI yang dianggap asimetris (default 0.05).
        
    Returns:
        dict berisi hasil pengecekan asimetri.
    """
    diff = abs(ai_left - ai_right)
    is_asym = diff >= threshold
    
    if is_asym:
        note = (f"[PERINGATAN] Terdeteksi asimetri struktural signifikan (delta AI = {diff:.4f}). "
                f"Perbedaan antara kaki kiri (AI={ai_left:.4f}) dan kanan (AI={ai_right:.4f}) "
                f"melebihi ambang batas {threshold}. Disarankan skrining lanjutan.")
    else:
        note = (f"[SIMETRI NORMAL] Simetri bilateral seimbang (delta AI = {diff:.4f}). "
                f"Kaki kiri (AI={ai_left:.4f}) dan kanan (AI={ai_right:.4f}) normal.")
    
    return {
        'difference': round(diff, 4),
        'is_asymmetric': is_asym,
        'note': note
    }


def get_recommendation(classification: str) -> dict:
    """
    Rekomendasi orthotics dan insole berdasarkan klasifikasi kelainan kaki.
    
    Args:
        classification: Hasil classify_arch ('High Arch', 'Normal', 'Flat Foot').
        
    Returns:
        dict berisi deskripsi klinis, tingkat risiko, rekomendasi insole,
        dan disclaimer medis.
    """
    DISCLAIMER = (
        "DISCLAIMER MEDIS: Sistem ini berfungsi murni sebagai alat SKRINING AWAL "
        "kesehatan dan edukasi, BUKAN diagnosis klinis definitif. Hasil analisis "
        "tidak menggantikan pemeriksaan fisik langsung oleh dokter spesialis ortopedi "
        "atau podiatris berlisensi."
    )
    
    recommendations = {
        'High Arch': {
            'class': 'High Arch (Pes Cavus)',
            'risk_level': 'Sedang (Peningkatan tekanan tumit & metatarsal)',
            'recommendation': (
                '- Gunakan Cushioned Insole (insol bantalan empuk) untuk mendistribusikan beban.\n'
                '- Pilih sepatu dengan shock absorption tinggi dan bantalan ekstra di sol tengah.\n'
                '- Pertimbangkan custom orthotic dengan lateral wedge support jika terdapat supitasi.\n'
                '- Lakukan peregangan (stretching) teratur pada plantar fascia dan otot achilles.'
            ),
            'description': (
                'Lengkungan kaki tinggi (pes cavus) menyebabkan berkurangnya area kontak midfoot '
                'dengan lantai, sehingga beban tubuh terkonsentrasi secara berlebihan pada tumit '
                'dan bola kaki depan (metatarsal heads). Kondisi ini meningkatkan kerentanan '
                'terhadap plantar fasciitis, metatarsalgia, dan ketidakstabilan pergelangan kaki.'
            ),
            'disclaimer': DISCLAIMER
        },
        'Normal': {
            'class': 'Normal Arch',
            'risk_level': 'Rendah (Biomekanik ideal)',
            'recommendation': (
                '- Lengkungan kaki dalam rentang normal dan fungsional.\n'
                '- Gunakan alas kaki standar yang nyaman dengan kontur arch support moderat.\n'
                '- Jaga kebugaran dan kekuatan otot kaki intrinsik secara konsisten.\n'
                '- Lakukan pemantauan berkala jika ada perubahan pola berjalan atau aktivitas fisik.'
            ),
            'description': (
                'Lengkungan telapak kaki normal memperlihatkan distribusi beban yang seimbang '
                'dan harmonis di antara forefoot, midfoot, dan rearfoot. Mekanisme windlass '
                'dan peredam kejut alami kaki berfungsi secara optimal.'
            ),
            'disclaimer': DISCLAIMER
        },
        'Flat Foot': {
            'class': 'Flat Foot (Pes Planus)',
            'risk_level': 'Sedang hingga Tinggi (Risiko overpronasi & misalignment)',
            'recommendation': (
                '- Gunakan insole dengan Medial Arch Support kokoh untuk menopang lengkungan.\n'
                '- Pilih sepatu bertipe Motion Control atau Stability Shoes untuk membatasi overpronasi.\n'
                '- Pertimbangkan konsultasi ortotik custom jika timbul nyeri pada pergelangan/lutut.\n'
                '- Lakukan senam kaki (towel curls, calf raises) guna memperkuat tendon tibialis posterior.'
            ),
            'description': (
                'Kaki datar (pes planus) ditandai dengan runtuhnya lengkungan medial longitudinal, '
                'sehingga area midfoot menyentuh tanah secara penuh. Hal ini sering memicu rotasi '
                'internal berlebih pada tibia (overpronation), nyeri lutut (patellofemoral pain), '
                'serta shin splints.'
            ),
            'disclaimer': DISCLAIMER
        },
        'Invalid': {
            'class': 'Tidak Terdefinisi',
            'risk_level': 'N/A',
            'recommendation': 'Data citra tidak valid untuk diekstraksi.',
            'description': 'Area kontur telapak kaki tidak terdeteksi.',
            'disclaimer': DISCLAIMER
        }
    }
    
    return recommendations.get(classification, recommendations['Invalid'])


def generate_report(filename: str, side: str, arch_index: float,
                    regions: dict) -> dict:
    """
    Generate ringkasan laporan analisis per individu citra.
    """
    classification = classify_arch(arch_index)
    recommendation = get_recommendation(classification)
    side_name = 'Kiri (Left)' if side.upper() == 'L' else 'Kanan (Right)'
    
    return {
        'filename': filename,
        'side': side,
        'side_name': side_name,
        'arch_index': arch_index,
        'classification': classification,
        'area_A': regions['A_area'],
        'area_B': regions['B_area'],
        'area_C': regions['C_area'],
        'total_area': regions['A_area'] + regions['B_area'] + regions['C_area'],
        'recommendation': recommendation,
    }
